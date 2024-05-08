"""
根据train中popularity直接进行推荐
直接取10个好了
"""
import numpy as np
import data_read as dr
import metric
import parser_init as parser
import random

args = parser.init_parser()


def cal_popularity(m_id, a_id, m_a_id):
    api_num = len(a_id)
    api_count_dic = dict()
    for api_list in m_a_id:
        for a in api_list:
            if a in api_count_dic.keys():
                api_count_dic[a] += 1
            else:
                api_count_dic[a] = 1

    # 存入list进行排序
    # 返回index(argsort)
    pop_list = np.zeros(api_num, dtype=np.int)
    for a in api_count_dic.keys():
        pop_list[a] = api_count_dic[a]

    sorted_pop_list = np.argsort(-pop_list)

    return sorted_pop_list


def cal_pop_dict(m_id, a_id, m_a_id):
    api_num = len(a_id)
    api_count_dic = dict()
    for api_list in m_a_id:
        for a in api_list:
            if a in api_count_dic.keys():
                api_count_dic[a] += 1
            else:
                api_count_dic[a] = 1

    return api_count_dic


if __name__ == "__main__":
    mashup_data_file_name = "recommend_mashup_api_id_more_than_3.txt"
    m_id, a_id, m_a_id = dr.read_mashup_api(mashup_data_file_name)

    mashup_num = len(m_id)
    api_num = len(a_id)

    # 随机获取根据train_scale
    print("初始化训练数据")
    train_mashup_list = random.sample(m_id, int(mashup_num * args.train_scale))
    test_mashup_list = [x for x in m_id if x not in train_mashup_list]
    print("train {}, test {}".format(len(train_mashup_list), len(test_mashup_list)))

    train_m_a_id = m_a_id[:]
    for test_index in test_mashup_list:
        train_m_a_id[test_index] = []

    sorted_api_pop_list = cal_popularity(m_id, a_id, train_m_a_id)
    k = 10
    k_sorted_api_pop_list = sorted_api_pop_list[0:k]

    whole_pop_list = []
    for i in range(len(test_mashup_list)):
        whole_pop_list.append(k_sorted_api_pop_list)

    average_hrk, hrk_s = metric.cal_average_hrk_metric(test_mashup_list, m_a_id, whole_pop_list, k)
    average_ndcg, ndcg_s = metric.cal_average_ndcg_k(test_mashup_list, m_a_id, whole_pop_list, k)
    mAP, ap_s = metric.cal_map(test_mashup_list, m_a_id, whole_pop_list, k)



