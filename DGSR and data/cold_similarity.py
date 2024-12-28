"""
利用description的feature求similarity
找到对应的相似服务
"""
import math
import numpy as np
import data_read as dr
import random
import time


def cos_sim(vec1, vec2):
    v_sum = 0
    for i in range(len(vec1)):
        v_sum = v_sum + vec1[i] * vec2[i]
    vec1_len = math.sqrt(sum([i * i for i in vec1]))
    vec2_len = math.sqrt(sum([i * i for i in vec2]))
    return v_sum / (vec1_len * vec2_len)


def dis_sim(vec1, vec2):
    v_sum = 0
    for i in range(len(vec1)):
        v_sum = v_sum + (vec1[i] - vec2[i]) * (vec1[i] - vec2[i])
    return 1 / v_sum


# 输入为训练index和测试index
# 输出结果为test_index为key，相似的train中index为list value的dic
# sim_num为保留的相似index数量
# sim_service_dic存放service真实序号
# sim_train_service_dic存放其在train_index中的位置信息
def cal_sim_service_from_train(train_index, test_index, features_vec, sim_num):
    t1 = time.time()
    sim_service_dic = dict()
    sim_train_service_dic = dict()
    # 计算相似度
    sim_matrix = np.zeros((len(test_index), len(train_index)))
    for i in range(len(test_index)):
        for j in range(len(train_index)):
            test_vec = features_vec[test_index[i]]
            train_vec = features_vec[train_index[j]]
            sim = dis_sim(test_vec, train_vec)
            sim_matrix[i][j] = sim
        print(i)
    print("sim计算完成")
    t2 = time.time()
    print("time:{}".format(t2-t1))
    # 排序并输出
    for i in range(len(sim_matrix)):
        single_test_index = test_index[i]
        sorted_index_list = np.argsort(-sim_matrix[i])

        sorted_train_index = [train_index[x] for x in sorted_index_list]
        sim_service_dic[single_test_index] = sorted_train_index[0:sim_num]
        sim_train_service_dic[single_test_index] = sorted_index_list[0:sim_num]
        print(single_test_index)
        print(sim_service_dic[single_test_index])
    return sim_service_dic, sim_train_service_dic


if __name__ == "__main__":
    t_m_bert_file_name = "recommend_mashup_bert_vec.txt"
    t_m_bert_feature = dr.read_mashup_bert_vec(t_m_bert_file_name, 1423, 768)

    train_mashup_list = random.sample(range(1423), int(1423 * 0.95))
    test_mashup_list = [x for x in range(1423) if x not in train_mashup_list]
    cal_sim_service_from_train(train_mashup_list, test_mashup_list, t_m_bert_feature, 3)
