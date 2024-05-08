"""
对MAIG进行处理，使用协同图神经来学习图结构信息
多层结合可以优化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ALCGNet import ALCGNet
import numpy as np


# attention network
# 用来算每一层的attention score
class MAAttentionLayer(nn.Module):
    """
    query: em, ea
    key: em, ena
    没准可以加上point-wise残差
    new_api_embeddings指从ALCG中学习到的api_embeddings
    """
    def __init__(self, input_dim, output_dim):
        super(MAAttentionLayer, self).__init__()
        self.q_linear = nn.Linear(input_dim, output_dim)
        self.k_linear = nn.Linear(input_dim, output_dim)

    def forward(self, mashup_embeddings, api_embeddings, new_api_embeddings, embedding_dim):
        combine_features = torch.cat([mashup_embeddings, api_embeddings], dim=0)
        combine_new_features = torch.cat([mashup_embeddings, new_api_embeddings], dim=0)
        query = self.q_linear(combine_features)
        key = self.k_linear(combine_new_features)
        up = torch.mm(query, key.t())  # [E,dim]*[dim,E]
        up = torch.divide(up, math.sqrt(embedding_dim))
        output = torch.sum(up, dim=1)  # [E,]

        return output


# 整合每层结果
class MLP(nn.Module):
    activation_classes = {'relu': nn.ReLU, 'tanh': nn.Tanh}

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, batch_norm=False,
                 init_last_layer_bias_to_zero=False, layer_norm=False, activation='relu'):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        assert not (self.batch_norm and self.layer_norm)

        self.layers = nn.Sequential()
        for i in range(self.num_layers + 1):
            n_in = self.input_size if i == 0 else self.hidden_size
            n_out = self.hidden_size if i < self.num_layers else self.output_size
            self.layers.add_module(f'{i}-Linear', nn.Linear(n_in, n_out))
            if i < self.num_layers:
                self.layers.add_module(f'{i}-Dropout', nn.Dropout(self.dropout))
                if self.batch_norm:
                    self.layers.add_module(f'{i}-BatchNorm1d', nn.BatchNorm1d(self.hidden_size))
                if self.layer_norm:
                    self.layers.add_module(f'{i}-LayerNorm', nn.LayerNorm(self.hidden_size))
                self.layers.add_module(f'{i}-{activation}', self.activation_classes[activation.lower()]())
        if init_last_layer_bias_to_zero:
            self.layers[-1].bias.data.fill_(0)

    def forward(self, input_x):
        return self.layers(input_x)


# 用来计算每层卷积结果
# 同时把每层的attention score计算出来
class GCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, device):
        super(GCLayer, self).__init__()
        self.device = device
        self.dropout = dropout
        self.linear0 = nn.Linear(input_dim, output_dim)
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.activation = nn.LeakyReLU()
        self.dropout_layer = nn.Dropout(dropout)
        self.attention_layer = MAAttentionLayer(input_dim, output_dim)

    def forward(self, mashup_num, A, self_loop, embedding_dim, former_embeddings, new_api_embeddings):
        # (A+I)*E + A*E pair_wise E

        A = A.to(self.device)
        self_loop = self_loop.to(self.device)
        AI = A + self_loop
        AI = AI.to(self.device)
        """
        A = A.to(self.device)
        AI = A.to(self.device)
        """

        part1 = self.linear0(torch.sparse.mm(AI, former_embeddings))  # [m+a,d]
        left = torch.sparse.mm(A, former_embeddings)  # [m+a,d]
        left = torch.mul(left, former_embeddings)  # [m+a,d]
        part2 = self.linear1(left)

        part = part1 + part2
        part = self.activation(part)
        part = self.dropout_layer(part)
        part = F.normalize(part)

        # attention score
        att_out = self.attention_layer(former_embeddings[0:mashup_num, ], former_embeddings[mashup_num: , ], new_api_embeddings, embedding_dim)
        return part, att_out


class MAIGNet(nn.Module):
    def __init__(self, mashup_num, api_num, embed_num, layer_size, drop_out, norm_adjacency_matrix,
                 a_norm_adjacency_matrix, prepare_embeddings, mlp_layer_num, mlp_layer_dim, mlp_drop_out, device,
                 reg, graph_drop, if_graph_drop=False):
        super(MAIGNet, self).__init__()
        self.mashup_num = mashup_num
        self.api_num = api_num
        self.embed_num = embed_num
        self.prepare_embeddings = prepare_embeddings
        self.new_api_embeddings = self.prepare_embeddings
        self.prepare_input_dim = len(prepare_embeddings[0])

        # 用来确定学习层数，格式为输出维数的list
        # [en, en, en..]
        self.layer_size = layer_size
        self.norm_adjacency_matrix = norm_adjacency_matrix
        self.device = device
        self.layer_num = len(layer_size)

        # [do,do,do..]
        self.drop_out = drop_out
        self.graph_drop = graph_drop
        self.if_graph_drop = if_graph_drop

        self.mashup_embeddings, self.api_embeddings = self.init_embeddings()
        self.g_layers = nn.ModuleList()
        # 初始化卷积层
        for in_dim, out_dim, do in zip(layer_size[:-1], layer_size[1:], drop_out):
            self.g_layers.append(GCLayer(in_dim, out_dim, do, self.device))

        # 把邻接矩阵转为tensor
        # 根据GCN需要把self-loop也算上
        self.a_norm_adjacency_matrix = a_norm_adjacency_matrix
        self.norm_adjacency_matrix = self.cal_adjacency_matrix_tensor()
        self.self_loop = self.cal_self_loop()
        # ALCG
        self.a_model = ALCGNet(self.api_num, self.prepare_input_dim, self.embed_num, self.a_norm_adjacency_matrix, self.device)

        # 求第一个layer的attention score
        self.mlp_layer_num = mlp_layer_num
        self.mlp_layer_dim = mlp_layer_dim
        self.mlp_drop_out = mlp_drop_out
        self.f_attention = MAAttentionLayer(self.embed_num, self.embed_num)
        self.mlp = MLP((len(layer_size) * self.embed_num), self.mlp_layer_dim, self.embed_num, self.mlp_layer_num, self.mlp_drop_out, batch_norm=True)
        self.mlp1 = MLP(self.embed_num, self.mlp_layer_dim, self.embed_num, self.mlp_layer_num, self.mlp_drop_out, batch_norm=True)

        # gate机制实现，bit-wise形式
        self.bit_gate = nn.Linear(self.embed_num * 2, 1)
        self.activation = nn.ReLU()

        # regular化参数
        self.reg = reg

    # 变成spare_tensor
    def cal_adjacency_matrix_tensor(self):
        sparse_mx = self.norm_adjacency_matrix.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    # 计算self loop
    def cal_self_loop(self):
        num = self.mashup_num + self.api_num
        i = torch.LongTensor([[k for k in range(0, num)], [j for j in range(0, num)]])
        val = torch.FloatTensor([1] * num)
        return torch.sparse.FloatTensor(i, val)

    # 对拉普拉斯对应的稀疏矩阵dropout
    # 总的还是把它部分mask掉
    def graph_dropout(self, spare_tensor, noise_shape):
        random_tensor = 1 - self.graph_drop
        random_tensor += torch.rand(noise_shape).to(self.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        """
        is_true = [x for x in dropout_mask if x == True]
        print(len(dropout_mask))
        print(len(is_true))
        """
        i = spare_tensor._indices()
        v = spare_tensor._values()
        i = i[:, dropout_mask]
        v = v[dropout_mask]
        out = torch.sparse.FloatTensor(i, v, spare_tensor.shape).to(self.device)
        return out * (1. / (1 - self.graph_drop))

    # bpr loss实现
    # - ln(sigmoid(yp - yn))
    # (考虑把末尾也实现了)
    # (考虑多项目k_bpr)
    def bpr_loss(self, mashup_embeddings, pos_api_embeddings, neg_api_embeddings):
        #  正负项用来算bpr
        # 为了方便计算，每个mashup采样一个pos和neg来进行计算
        pos_scores = torch.sum(torch.mul(mashup_embeddings, pos_api_embeddings), dim=1)
        neg_scores = torch.sum(torch.mul(mashup_embeddings, neg_api_embeddings), dim=1)
        maxi = nn.LogSigmoid()(pos_scores - neg_scores)
        mf_loss = -1 * torch.mean(maxi)
        return mf_loss

    # whole bpr loss实现
    # - sigmoid(yp - yn) + 后面的regular
    # 对pos和neg和本体进行norm
    def whole_bpr_loss(self, mashup_embeddings, pos_api_embeddings, neg_api_embeddings):
        #  正负项用来算bpr
        # 为了方便计算，每个mashup采样一个pos和neg来进行计算
        pos_scores = torch.sum(torch.mul(mashup_embeddings, pos_api_embeddings), dim=1)
        neg_scores = torch.sum(torch.mul(mashup_embeddings, neg_api_embeddings), dim=1)
        maxi = nn.LogSigmoid()(pos_scores - neg_scores)
        mf_loss = -1 * torch.mean(maxi)

        regular = (torch.norm(mashup_embeddings) ** 2
                       + torch.norm(pos_api_embeddings) ** 2
                       + torch.norm(neg_api_embeddings) ** 2) / 2
        emb_loss = self.reg * regular
        return mf_loss + emb_loss

    # 完全sample尝试
    # (或者单个正例多个负例)
    # mashup_index为对应sample的m序号，mashup_embeddings为按mashup_index顺序排的embedding，别用错了
    # 从字典型的pos_dic和neg_dic中取出多个正负例
    def multi_bpr_loss(self, mashup_index, mashup_embeddings, api_embeddings, pos_dic, neg_dic):
        loss = None
        for i in range(len(mashup_index)):
            s_mashup_embedding = mashup_embeddings[i]
            s_mashup_index = mashup_index[i]
            s_sample_pos_list = pos_dic[s_mashup_index]
            s_sample_neg_list = neg_dic[s_mashup_index]

            # [pos_sample_num, embed_dim] -> [pos_sample_num * neg_sample_num, embed_dim]
            # [0, 1...] - > [0,0,1,1...]
            # [neg_sample_num, embed_dim] -> [neg_sample_num * neg_sample_num, embed_dim]
            # [0, 1...] - > [0,1,...,0,1...]
            p_n_index = []
            for p in s_sample_pos_list:
                p_n_index.extend([p] * len(s_sample_neg_list))
            n_p_index = s_sample_neg_list * len(s_sample_pos_list)
            pos_embeddings = api_embeddings[p_n_index]
            neg_embeddings = api_embeddings[n_p_index]

            m_pos_neg_e = s_mashup_embedding.repeat(len(s_sample_pos_list) * len(s_sample_neg_list), 1)
            pos_scores = torch.sum(torch.mul(m_pos_neg_e, pos_embeddings), dim=1)  # [p*n, 1]
            neg_scores = torch.sum(torch.mul(m_pos_neg_e, neg_embeddings), dim=1)  # [p*n, 1]
            maxi = nn.LogSigmoid()(pos_scores - neg_scores)
            mf_loss = -1 * torch.mean(maxi)
            if loss is None:
                loss = mf_loss
            else:
                loss = mf_loss + loss
        return loss

    # 加上后面的l2正则参数
    def multi_whole_bpr_loss(self, mashup_index, mashup_embeddings, api_embeddings, pos_dic, neg_dic):
        loss = None
        r_loss = None
        for i in range(len(mashup_index)):
            s_mashup_embedding = mashup_embeddings[i]
            s_mashup_index = mashup_index[i]
            s_sample_pos_list = pos_dic[s_mashup_index]
            s_sample_neg_list = neg_dic[s_mashup_index]

            used_sample_pos = api_embeddings[s_sample_pos_list]
            used_sample_neg = api_embeddings[s_sample_neg_list]

            # [pos_sample_num, embed_dim] -> [pos_sample_num * neg_sample_num, embed_dim]
            # [0, 1...] - > [0,0,1,1...]
            # [neg_sample_num, embed_dim] -> [neg_sample_num * neg_sample_num, embed_dim]
            # [0, 1...] - > [0,1,...,0,1...]
            p_n_index = []
            for p in s_sample_pos_list:
                p_n_index.extend([p] * len(s_sample_neg_list))
            n_p_index = s_sample_neg_list * len(s_sample_pos_list)
            pos_embeddings = api_embeddings[p_n_index]
            neg_embeddings = api_embeddings[n_p_index]

            m_pos_neg_e = s_mashup_embedding.repeat(len(s_sample_pos_list) * len(s_sample_neg_list), 1)
            pos_scores = torch.sum(torch.mul(m_pos_neg_e, pos_embeddings), dim=1)  # [p*n, 1]
            neg_scores = torch.sum(torch.mul(m_pos_neg_e, neg_embeddings), dim=1)  # [p*n, 1]
            maxi = nn.LogSigmoid()(pos_scores - neg_scores)
            mf_loss = -1 * torch.mean(maxi)
            if loss is None:
                loss = mf_loss
            else:
                loss = mf_loss + loss

            # 求范式结果
            regular = (torch.norm(mashup_embeddings) ** 2
                       + torch.norm(used_sample_pos) ** 2
                       + torch.norm(used_sample_neg) ** 2) / 2
            emb_loss = self.reg * regular
            if r_loss is None:
                r_loss = emb_loss
            else:
                r_loss = emb_loss + r_loss
        return r_loss + loss

    # 初始化embeddings，作为可训练参数
    def init_embeddings(self):
        # xavier init均匀分布
        initializer = nn.init.xavier_uniform_
        mashup_embedding = nn.Parameter(initializer(torch.empty(self.mashup_num, self.embed_num)))
        api_embedding = nn.Parameter(initializer(torch.empty(self.api_num, self.embed_num)))
        return mashup_embedding, api_embedding

    # 计算api对mashup的得分信息
    # test用，算指标
    def get_score(self, mashup_embeddings, item_embeddings):
        return torch.matmul(mashup_embeddings, item_embeddings.t())  # [m,a]

    # forward包括layer层数值计算，attention连接和最终的gate连接
    def forward(self, mashup_index, api_index):
        A = self.norm_adjacency_matrix
        if self.if_graph_drop:
            A = self.graph_dropout(A, A._nnz())
        feature_embeddings = torch.cat([self.mashup_embeddings, self.api_embeddings], dim=0)  # E

        # 从ALCG中获取embeddings
        self.new_api_embeddings = self.a_model(self.prepare_embeddings)
        first_att_out = self.f_attention(self.mashup_embeddings, self.api_embeddings, self.new_api_embeddings, self.embed_num)  # [E,1]
        all_features = [feature_embeddings]
        att_outs = [first_att_out]

        feature_out = feature_embeddings
        for g_layer in self.g_layers:
            feature_out, att_out = g_layer(self.mashup_num, A, self.self_loop, self.embed_num, feature_out, self.new_api_embeddings)
            all_features.append(feature_out)
            att_outs.append(att_out)

        # attention score softmax后相乘
        att_outs = torch.stack(att_outs)  # [L+1,E]
        a_s = F.softmax(att_outs, dim=0)  # [L+1,E]
        a_s = torch.unsqueeze(a_s, 2)  # [L+1,E,1]
        a_s = torch.repeat_interleave(a_s, repeats=self.embed_num, dim=2)  # [L+1,E,dim]

        all_features = torch.stack(all_features)  # [L+1,E,dim]
        a_a_features = torch.mul(a_s, all_features)  # [L+1,E,dim]
        combine_features = []  # 对应tensor连接
        for i in range(a_a_features.size()[1]):
            combine_features.append(torch.cat([a_a_features[x][i] for x in range(a_a_features.size()[0])], dim=0))
        combine_features = torch.stack(combine_features)  # [E,dim*(L+1)]

        # 过MLP(带BN) [E,dim*(L+1)] -> [E,dim]
        check_features = self.mlp(combine_features)

        # 获取mashup和api信息
        whole_item_index = [(x + self.mashup_num) for x in api_index]
        mashups = check_features[mashup_index]
        apis = check_features[whole_item_index]

        # gate机制结合apis和new_api_embeddings
        # (要不要全部gate一下呢)
        n_apis = self.new_api_embeddings[api_index]
        combine_apis = torch.cat([n_apis, apis], dim=1)
        gate_score = self.bit_gate(combine_apis)
        api_output = gate_score * apis + (1 - gate_score) * n_apis
        api_output = self.mlp1(api_output)

        return mashups, api_output




