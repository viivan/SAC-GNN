"""
读取数据信息
Mashup调用api情况
"""
import numpy as np
import lm_features as lm


# mashup_id是按照顺序排列的，直接整就好了
# 返回mashup的id list，api的id list和mashup对应的调用api的结果
# mashup_id api_id0 api_id1...
def read_mashup_api(file_name):
    m_id = []
    a_id = []
    m_a_list = []
    with open(file_name, "r", encoding="utf-8") as f:
        for l in f.readlines():
            line_list = l.strip().split(" ")

            mashup_id = line_list[0]
            m_id.append(mashup_id)

            api_id_list = line_list[1:]
            m_a_list.append(api_id_list)
            for a in api_id_list:
                if a not in a_id:
                    a_id.append(a)
    print("获取mashup调用信息")
    print("mashup num:{}".format(len(m_id)))
    print("api num:{}".format(len(a_id)))

    # 为了方便处理全转为int
    m_id = [int(x) for x in m_id]
    a_id = [int(x) for x in a_id]
    for i in range(len(m_a_list)):
        m_a_list[i] = [int(x) for x in m_a_list[i]]

    return m_id, a_id, m_a_list


# 读取api tag调用信息，每行为对应id的api的share tag的api id
# api_id api_id1 api_id1...
def read_api(file_name):
    a_id = []
    a_a_id = []
    with open(file_name, "r", encoding="utf-8") as f:
        for l in f.readlines():
            line_list = l.strip().split(" ")
            api_id = line_list[0]
            a_id.append(api_id)
            api_share_list = line_list[1:]
            a_a_id.append(api_share_list)
    print("获取api关联信息")
    print("api num:{}".format(len(a_id)))

    # 为了方便处理全转为int
    a_id = [int(x) for x in a_id]
    for i in range(len(a_a_id)):
        a_a_id[i] = [int(x) for x in a_a_id[i]]
    return a_id, a_a_id


# 读取api分词后的描述文档
# word0 word1 word2...
def read_api_divide_description(file_name):
    divide_description = []
    with open(file_name, "r", encoding="utf-8") as f:
        for l in f.readlines():
            line_list = l.strip().split(" ")
            divide_description.append(line_list)
    print("获取api分词文档")
    print("api num:{}".format(len(divide_description)))
    return divide_description


# 读取w2v持久化后的向量结果
def read_api_w2v_vec(file_name, api_num, vec_dim=300):
    # 直接用numpy读
    features = np.fromfile(file_name)
    w2v_features = features.reshape((api_num, vec_dim))
    print("读取word2vec计算向量成功")
    print("api num:{}".format(len(w2v_features)))
    return w2v_features


# 读取bert持久化后的向量结果
def read_api_bert_vec(file_name, api_num, vec_dim=768):
    # 直接用numpy读
    features = np.fromfile(file_name, dtype=np.float32)
    bert_features = features.reshape((api_num, vec_dim))
    print(bert_features.shape)
    print("读取BERT计算向量成功")
    print("api num:{}".format(len(bert_features)))
    return bert_features


# 读取mashup分词后的描述文档
# word0 word1 word2...
def read_mashup_divide_description(file_name):
    divide_description = []
    with open(file_name, "r", encoding="utf-8") as f:
        for l in f.readlines():
            line_list = l.strip().split(" ")
            divide_description.append(line_list)
    print("获取mashup分词文档")
    print("mashup num:{}".format(len(divide_description)))
    return divide_description


# 读取bert持久化后的向量结果
def read_mashup_bert_vec(file_name, mashup_num, vec_dim=768):
    # 直接用numpy读
    features = np.fromfile(file_name, dtype=np.float32)
    bert_features = features.reshape((mashup_num, vec_dim))
    print(bert_features.shape)
    print("读取BERT计算向量成功")
    print("mashup num:{}".format(len(bert_features)))
    return bert_features


if __name__ == "__main__":
    """
    t_m_a_file_name = "recommend_mashup_api_id_more_than_3.txt"
    read_mashup_api(t_m_a_file_name)

    t_a_s_file_name = "recommend_api_share_tag_id_more_than_3.txt"
    read_api(t_a_s_file_name)

    t_a_d_file_name = "recommend_api_divide_description.txt"
    read_api_divide_description(t_a_d_file_name)
    """

    """
    t_a_w2v_file_name = "recommend_api_w2v_vec.txt"
    t_w2v_features = read_api_w2v_vec(t_a_w2v_file_name, 1032, 300)
    print(t_w2v_features.shape)
    """

    t_a_bert_file_name = "recommend_api_bert_vec.txt"
    # t_bert_features = read_api_bert_vec(t_a_bert_file_name, 1032, 768)

    t_m_divide_description_file_name = "recommend_mashup_divide_description.txt"
    t_m_bert_file_name = "recommend_mashup_bert_vec.txt"
    t_m_des = read_mashup_divide_description(t_m_divide_description_file_name)
    t_m_bert_feature = read_mashup_bert_vec(t_m_bert_file_name, 1423, 768)
