"""
主要训练过程
模拟全新mashup需求的输入
"""
import numpy as np
import torch
import torch.nn as nn
import parser_init as parser
import data_read as dr
import model_test
import random
import graph_calculate as gc
import MAIGNet
import time
import lm_features as lm
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import StepLR
import cold_similarity as c_s

args = parser.init_parser()


# 训练方法，主要就是调用model算下loss
# args为参数设置
def train(mashup_list, api_list, sample_num, m_a_id, model, optimizer, scheduler=None):
    pos_api_index, neg_api_index = model_test.sample_pos_neg_api(mashup_list, api_list, sample_num, m_a_id)

    whole_train_api_index = pos_api_index + neg_api_index
    mashup_embeddings, api_embeddings = model(mashup_list, whole_train_api_index)

    # 按照pos和neg分离
    pos_num = len(pos_api_index)
    pos_api_embeddings = api_embeddings[:pos_num]
    neg_api_embeddings = api_embeddings[pos_num:]

    mf_loss = 0
    # 计算loss
    if args.loss_func == 'sample_bpr':
        # 单次随机采样bpr
        mf_loss = model.bpr_loss(mashup_embeddings, pos_api_embeddings, neg_api_embeddings)
    elif args.loss_func == 'whole_bpr':
        # 全种bpr
        mf_loss = model.whole_bpr_loss(mashup_embeddings, pos_api_embeddings, neg_api_embeddings)
    optimizer.zero_grad()
    # loss也是tensor形式的
    mf_loss.backward()
    optimizer.step()
    if scheduler is not None:
        print("学习率：%f" % (optimizer.param_groups[0]['lr']))
        scheduler.step()

    return mf_loss


# 多重训练
def multi_train(mashup_list, api_list, pos_sample_num, neg_sample_num, m_a_id, model, optimizer, scheduler=None):
    m_pos_dic, m_neg_dic = model_test.sample_multi_pos_neg_api(mashup_list, api_list, pos_sample_num, neg_sample_num,
                                                               m_a_id)
    mashup_embeddings, api_embeddings = model(mashup_list, api_list)

    loss = 0
    # 多正负例loss计算
    if args.loss_func == 'sample_bpr':
        loss = model.multi_bpr_loss(mashup_list, mashup_embeddings, api_embeddings, m_pos_dic, m_neg_dic)
    elif args.loss_func == 'whole_bpr':
        loss = model.multi_whole_bpr_loss(mashup_list, mashup_embeddings, api_embeddings, m_pos_dic, m_neg_dic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        print("学习率：%f" % (optimizer.param_groups[0]['lr']))
        scheduler.step()

    return loss


if __name__ == "__main__":
    # 每10次进行一个test，loss收集最后绘图
    # 读取文件信息
    mashup_data_file_name = "recommend_mashup_api_id_more_than_3.txt"
    api_data_file_name = "recommend_api_share_tag_id_more_than_3.txt"
    api_desc_divide_file_name = "recommend_api_divide_description.txt"
    m_id, a_id, m_a_id = dr.read_mashup_api(mashup_data_file_name)
    s_a_id, a_a_id = dr.read_api(api_data_file_name)

    api_w2v_vec_file_name = "recommend_api_w2v_vec.txt"
    api_bert_vec_file_name = "recommend_api_bert_vec.txt"

    mashup_bert_vec_file_name = "recommend_mashup_bert_vec.txt"

    # 计算邻接矩阵
    mashup_num = len(m_id)
    api_num = len(a_id)
    MAIG_adj_matrix, MAIG_norm_adj_matrix = gc.create_scipy_adjacency_matrix_MAIG(m_id, a_id, m_a_id)
    ALCG_adj_matrix, ALCG_norm_adj_matrix = gc.create_scipy_adjacency_matrix_ALCG_no_type(a_id, a_a_id)

    # 随机获取根据train_scale
    print("初始化训练数据")
    train_mashup_list = random.sample(m_id, int(mashup_num * args.train_scale))
    test_mashup_list = [x for x in m_id if x not in train_mashup_list]
    print("train {}, test {}".format(len(train_mashup_list), len(test_mashup_list)))

    # 找到相似mashup
    t_m_bert_feature = dr.read_mashup_bert_vec(mashup_bert_vec_file_name, mashup_num, args.prepare_embed_dim_bert)
    sim_service_dic, sim_train_service_dic = c_s.cal_sim_service_from_train(train_mashup_list, test_mashup_list, t_m_bert_feature, args.cold_sim_num)

    t_p_e = None
    if args.prepare_method == "random":
        t_p_e = torch.rand([api_num, args.embed_num], out=None)  # 测试用，真实使用的话利用language model
    elif args.prepare_method == "word2vec":
        """
        print("利用word2vec初始化api特征")
        t_p_e = torch.tensor(lm.cal_feature_vec_from_word2vec(api_desc_divide_file_name)).to(torch.float32)
        print("完成向量创建")
        """
        print("读取利用word2vec初始化api特征")
        t_p_e = torch.tensor(dr.read_api_w2v_vec(api_w2v_vec_file_name, api_num, args.prepare_embed_dim_w2v)).to(
            torch.float32)
        print("完成向量读取")
    elif args.prepare_method == "bert":
        print("读取利用BERT初始化api特征")
        t_p_e = torch.tensor(dr.read_api_bert_vec(api_bert_vec_file_name, api_num, args.prepare_embed_dim_bert)).to(
            torch.float32)
        print("完成向量读取")

    model = MAIGNet.MAIGNet(mashup_num, api_num, args.embed_num, args.layer_size, args.drop_out,
                            MAIG_norm_adj_matrix,
                            ALCG_norm_adj_matrix, t_p_e, args.mlp_layer_num, args.mlp_layer_dim, args.drop_out_fc,
                            args.device, args.regs[0], args.graph_dropout, if_graph_drop=True)
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)
    schedule = StepLR(optimizer, step_size=3000, gamma=0.1)

    max_average_hrk, max_average_ndcg = 0, 0
    for i in range(args.epoch):
        # 先sample对应的正负api
        t1 = time.time()
        # loss = train(train_mashup_list, a_id, args.sample_num, m_a_id, model, optimizer)
        loss = multi_train(train_mashup_list, a_id, args.pos_sample_num, args.neg_sample_num, m_a_id, model, optimizer,
                           schedule)
        t2 = time.time()
        print("epoch {}, loss {}, time {}s".format(i, loss, round((t2 - t1), 1)))

        if i != 0 and i % 20 == 0:
            print("----------------------------------------------------------")
            t5 = time.time()
            print("train metric:")
            model_test.test(model, train_mashup_list, a_id, m_a_id, args)
            t6 = time.time()
            print("time {}".format(round(t6 - t5, 1)))

            t3 = time.time()
            print("test metric:")
            average_hrk, average_ndcg = model_test.cold_test(model, train_mashup_list, test_mashup_list, a_id, m_a_id, sim_service_dic, sim_train_service_dic, args)
            if average_ndcg > max_average_ndcg:
                max_average_ndcg = average_ndcg
            if average_hrk > max_average_hrk:
                max_average_hrk = average_hrk
            t4 = time.time()
            print("time {}".format(round(t4 - t3, 1)))
            print("----------------------------------------------------------")

        if i != 0 and i % 100 == 0:
            print("max hr:{}".format(max_average_hrk))
            print("max ndcg:{}".format(max_average_ndcg))
            print("----------------------------------------------------------")
