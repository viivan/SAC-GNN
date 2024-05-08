"""
测试相关方法
包括正样本和负样本的采样sample
train和test数据的挣扎
"""
import numpy as np
import metric
import heapq
import multiprocessing
from functools import partial
import time
import torch


# 采样样本
def sample_api_to_num(api_list, num):
    result_list = []
    n = len(api_list)
    if n <= num:
        num = n
    while True:
        if len(result_list) == num:
            break
        pos_id = np.random.randint(low=0, high=n, size=1)[0]
        api_index = api_list[pos_id]
        if api_index not in result_list:
            result_list.append(api_index)
    return result_list


# sample正负样本
# batch控制是否要用小批量
# m_a_id_list存放mashup调用信息
# 直接sample 1个
def sample_pos_neg_api(mashup_index, api_index, sample_num, m_a_id_list, batch_size=200, if_batch=False):
    if not if_batch:
        # 全体使用来进行训练
        # 为每个mashup计算pos与neg的api_index
        whole_pos_list = []
        whole_neg_list = []
        for m_index in mashup_index:
            pos_list = m_a_id_list[m_index]
            neg_list = [x for x in api_index if x not in pos_list]
            pos_sample_list = sample_api_to_num(pos_list, sample_num)
            neg_sample_list = sample_api_to_num(neg_list, sample_num)
            whole_pos_list.extend(pos_sample_list)
            whole_neg_list.extend(neg_sample_list)
        # print(whole_neg_list)
        return whole_pos_list, whole_neg_list


# sample多个正负样本
# 每个mashup_index去找pos_sample_num个正例，neg_sample_num个负例
# 形成两个dic格式，key为对应的mashup_index好了
def sample_multi_pos_neg_api(mashup_index, api_index, pos_sample_num, neg_sample_num, m_a_id_list, batch_size=200, if_batch=False):
    if not if_batch:
        # 全体使用来进行训练
        # 为每个mashup计算pos与neg的api_index
        m_pos_dic = dict()
        m_neg_dic = dict()
        for m_index in mashup_index:
            pos_list = m_a_id_list[m_index]
            neg_list = [x for x in api_index if x not in pos_list]
            pos_sample_list = sample_api_to_num(pos_list, pos_sample_num)
            neg_sample_list = sample_api_to_num(neg_list, neg_sample_num)
            m_pos_dic[m_index] = pos_sample_list
            m_neg_dic[m_index] = neg_sample_list
        return m_pos_dic, m_neg_dic


# 排序并获取前k位
def sort_api_score_for_top(sort_k, r_s):
    i_s = {}
    for i in range(len(r_s)):
        i_s[i] = r_s[i]
    recommend_index = heapq.nlargest(sort_k, i_s, key=i_s.get)
    return recommend_index


# 用来计算所有的metric结果，方便多线程
def cal_metric(test_mashup_index, m_a_id_list, recommend_score, args):
    hr_k = args.hr_k  # hit ratio
    ndcg_k = args.ndcg_k  # normalized DCG
    map_k = args.map_k

    # 在外部把排好的送进去
    # 选取最大值
    len_list = [len(x) for x in m_a_id_list]
    sort_k = max(hr_k, ndcg_k, map_k, max(len_list))

    t1 = time.time()
    # 改为并行
    pool = multiprocessing.Pool(args.cpu_core)
    func = partial(sort_api_score_for_top, sort_k)

    recommend_score = recommend_score.detach()
    sorted_recommend_index = pool.map(func, recommend_score)
    # print(sorted_recommend_index)
    pool.close()
    pool.join()
    t2 = time.time()
    # print(t2-t1, "大概并行")

    # metric.cal_average_recall_metric(test_mashup_index, m_a_id_list, sorted_recommend_index)
    average_hrk, hrk_s = metric.cal_average_hrk_metric(test_mashup_index, m_a_id_list, sorted_recommend_index, hr_k)
    average_ndcg, ndcg_s = metric.cal_average_ndcg_k(test_mashup_index, m_a_id_list, sorted_recommend_index, ndcg_k)
    mAP, ap_s = metric.cal_map(test_mashup_index, m_a_id_list, sorted_recommend_index, map_k)

    return average_hrk, average_ndcg


# 测试方法
# api_index包含所有的api，用来在测试时找到所有api中推荐的
def test(model, test_mashup_index, api_index, m_a_id_list, args):
    mashup_features, api_features = model(test_mashup_index, api_index)
    recommend_score = model.get_score(mashup_features, api_features)  # [t_m,a]

    recommend_score = recommend_score.to("cpu")
    # 可以考虑把结果持久化一下
    average_hrk, average_ndcg = cal_metric(test_mashup_index, m_a_id_list, recommend_score, args)
    return average_hrk, average_ndcg


# cold测试
# 情景是完全没有历史记录的服务需求出现
# test和train对应的vec求相似度(提前求好)
# 利用相似度求出对应的n个train中的mashup求mean——pool
# 输出结果再算test
# sim_service_dic存放service真实序号
# sim_train_service_dic存放其在train_index中的位置信息
def cold_test(model, train_mashup_index, test_mashup_index, api_index, m_a_id_list, sim_service_dic, sim_train_service_dic, args):
    cold_sim_num = args.cold_sim_num  # 相似服务数量
    mashup_features, api_features = model(train_mashup_index, api_index)

    # 通过mashup_features和sim_service来求对应的test_features
    test_sim_feature_list = []  # test_num个
    for t_i in test_mashup_index:
        # 把对应的结果mean-pooling
        single_sim_train_index = sim_train_service_dic[t_i]
        single_sim_train_features = mashup_features[single_sim_train_index]  # cold_sim_num个
        single_mean_test_feature = torch.mean(single_sim_train_features, dim=0)
        single_mean_test_feature = single_mean_test_feature.view(1, single_mean_test_feature.shape[0])
        test_sim_feature_list.append(single_mean_test_feature)
    test_features = torch.cat(test_sim_feature_list, dim=0)

    recommend_score = model.get_score(test_features, api_features)  # [t_m,a]

    recommend_score = recommend_score.to("cpu")
    # 可以考虑把结果持久化一下
    average_hrk, average_ndcg = cal_metric(test_mashup_index, m_a_id_list, recommend_score, args)
    return average_hrk, average_ndcg


# 对测试集测试，将其在训练集中的invoke信息剔除后，计算推荐的效果
def test_for_test(model, test_mashup_index, api_index, test_m_a_id_list, train_m_a_id_list, args):
    mashup_features, api_features = model(test_mashup_index, api_index)
    recommend_score = model.get_score(mashup_features, api_features)  # [t_m,a]

    recommend_score = recommend_score.detach()
    recommend_score = recommend_score.to("cpu")
    # 将其训练集中的invoke都置为0，这样不会有高顺位
    for i in range(len(test_mashup_index)):
        single_index = test_mashup_index[i]
        for j in train_m_a_id_list[single_index]:
            recommend_score[i][j] = 0
    average_hrk, average_ndcg = cal_metric(test_mashup_index, test_m_a_id_list, recommend_score, args)
    return average_hrk, average_ndcg




