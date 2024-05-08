"""
评判指标
"""
import numpy as np
import math


# 召回率,api推荐不考虑顺序，直接对上几个算几个
# (r∩g)/n
def recall(recommend_sort_list, ground_truth):
    n = len(ground_truth)
    r_list = recommend_sort_list[:n]
    intersection = 0
    for g in ground_truth:
        if g in r_list:
            intersection = intersection + 1
    return intersection / n


# 计算平均recall
def cal_average_recall_metric(test_mashup_index, m_a_id_list, sorted_recommend_score):
    # 对每个选入的mashup进行计算，recommend_score对应行
    recalls = []
    for i in range(len(test_mashup_index)):
        m_index = test_mashup_index[i]
        ground_result = m_a_id_list[m_index]
        r_s = sorted_recommend_score[i]
        s_recall = recall(r_s, ground_result)
        recalls.append(s_recall)
    average_recall = sum(recalls) / len(recalls)
    print("average recall:{}".format(average_recall))
    return average_recall, recalls


# dcg实现
# 和推荐位置相关
def dcg(recommend_sort_list, ground_truth, k):
    whole_dcg = 0
    if k > len(recommend_sort_list):
        k = len(recommend_sort_list)
    for i in range(k):
        r = recommend_sort_list[i]
        ri = 0
        if r in ground_truth:
            ri = 1
        s_dec = (math.pow(2, ri) - 1) / (math.log2(1 + (i+1)))
        whole_dcg += s_dec
    return whole_dcg


# ndcg@指标实现
def ndcg(recommend_sort_list, ground_truth, k):
    r_dcg = dcg(recommend_sort_list, ground_truth, k)
    n_dcg = dcg(ground_truth, ground_truth, k)
    n_dcg = r_dcg / n_dcg
    return n_dcg


# 计算平均ndcg@k结果
def cal_average_ndcg_k(test_mashup_index, m_a_id_list, sorted_recommend_score, k):
    # 对每个选入的mashup进行计算，recommend_score对应行
    ndcg_s = []
    for i in range(len(test_mashup_index)):
        m_index = test_mashup_index[i]
        ground_result = m_a_id_list[m_index]
        r_s = sorted_recommend_score[i]
        s_ndcg = ndcg(r_s, ground_result, k)
        ndcg_s.append(s_ndcg)
    average_ndcg = sum(ndcg_s) / len(ndcg_s)
    print("average ndcg:{}".format(average_ndcg))
    return average_ndcg, ndcg_s


# hr@指标实现
# k个中有几个中了
# hits / len(gt)
def hr_k(recommend_sort_list, ground_truth, k):
    n = len(ground_truth)
    if k <= len(recommend_sort_list):
        r_list = recommend_sort_list[:k]
    else:
        r_list = recommend_sort_list[:len(recommend_sort_list)]
    intersection = 0
    for g in ground_truth:
        if g in r_list:
            intersection = intersection + 1
    return intersection / min([n, k])


# 计算平均hr@k
def cal_average_hrk_metric(test_mashup_index, m_a_id_list, sorted_recommend_score, k):
    # 对每个选入的mashup进行计算，recommend_score对应行
    hrk_s = []
    for i in range(len(test_mashup_index)):
        m_index = test_mashup_index[i]
        ground_result = m_a_id_list[m_index]
        r_s = sorted_recommend_score[i]
        s_hrk = hr_k(r_s, ground_result, k)
        hrk_s.append(s_hrk)
    average_hrk = sum(hrk_s) / len(hrk_s)
    print("average hr@k:{}".format(average_hrk))
    return average_hrk, hrk_s


# 计算ap
def cal_ap(recommend_sort_list, ground_truth, k):
    hits = 0
    sum_p = 0
    if k > len(recommend_sort_list):
        k = len(recommend_sort_list)
    for n in range(k):
        if recommend_sort_list[n] in ground_truth:
            hits += 1
            sum_p += hits / (n + 1.0)
    if hits > 0:
        return sum_p / min(len(ground_truth), k)
    else:
        return 0


# 计算map(mean average precision)
# 先求precision和ap，最后mean
def cal_map(test_mashup_index, m_a_id_list, sorted_recommend_score, k):
    ap_s = []
    for i in range(len(test_mashup_index)):
        m_index = test_mashup_index[i]
        ground_result = m_a_id_list[m_index]
        r_s = sorted_recommend_score[i]
        s_ap = cal_ap(r_s, ground_result, k)
        # print(s_ap)
        ap_s.append(s_ap)
    mAP = sum(ap_s) / len(ap_s)
    print("mAP:{}".format(mAP))
    return mAP, ap_s