"""
对ALCG进行实现
初始embedding利用外部LM初始化，中间两层GCN进行学习
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 图卷积层，可叠加
class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim).to(device)

    def forward(self, norm_adj, features, device):
        norm_adj = norm_adj.to(device)
        features = features.to(device)
        f = torch.sparse.mm(norm_adj, features)
        out = self.linear(f)
        return out
                                

class ALCGNet(nn.Module):
    def __init__(self, api_num, input_dim, embed_num, norm_adjacency_matrix, device):
        super(ALCGNet, self).__init__()
        self.api_num = api_num
        self.input_dim = input_dim
        self.embed_num = embed_num
        self.device = device
        self.norm_adjacency_matrix = norm_adjacency_matrix

        # 多一个全连接把文档向量转为对应维度
        self.linear_start = nn.Linear(input_dim, embed_num)
        # 初始化卷积层
        self.g_layer0 = GCNLayer(embed_num, 64, device).to(device)
        self.g_layer1 = GCNLayer(64, embed_num, device).to(device)
        self.norm_adj = self.cal_adjacency_matrix_tensor()
        self.norm_adj = self.norm_adj + self.cal_self_loop()  # A+I
        self.norm_adj.to(self.device)

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
        num = self.api_num
        i = torch.LongTensor([[k for k in range(0, num)], [j for j in range(0, num)]])
        val = torch.FloatTensor([1] * num)
        return torch.sparse.FloatTensor(i, val)

    def forward(self, features):
        features = features.to(self.device)
        features = self.linear_start(features)
        h = F.relu(self.g_layer0(self.norm_adj, features, self.device))
        output = self.g_layer1(self.norm_adj, h, self.device)
        return output


