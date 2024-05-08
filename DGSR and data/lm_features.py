"""
调用语言模型对api描述进行向量表示
包括分词操作,tf-idf来做权重连接
做好持久化
"""
import data_read as dr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from gensim.models import word2vec
from gensim.models import KeyedVectors
import numpy as np
import torch
from bert_serving.client import BertClient


# 计算tf-idf值
# matrix中元素为单独单词
def cal_tf_idf(word_divide_matrix):
    corpus = []
    for w_d in word_divide_matrix:
        corpus.append(" ".join(w_d))
    vector = CountVectorizer()  # 获取词汇解释器
    transformer = TfidfTransformer()  # tf-idf计算
    tf_idf = transformer.fit_transform(vector.fit_transform(corpus))
    word_dic = vector.get_feature_names()  # 获取词汇矩阵
    weight = tf_idf.toarray()  # 每个文档对于每个词的tf_idf值

    return word_dic, weight


# 读取word2vec的模型结果
def read_pre_trained_word2vec(save_path=r"E:\研究生\word2vec\GoogleNews_vec.bin"):
    print("加载模型文件{}".format(save_path))
    model = KeyedVectors.load_word2vec_format(save_path, binary=True)
    print("加载完毕")
    return model


# 将tf_idf给的weight转为(word,tf_idf)的元组集合
def convert_weight_into_tuple(word_dic, weight):
    word_tf_tuple = []
    for s_weight in weight:
        s_w_tt = []
        for i in range(len(s_weight)):
            if s_weight[i] != 0:
                s_w_tt.append((word_dic[i], s_weight[i]))
        word_tf_tuple.append(s_w_tt)
    return word_tf_tuple


# 利用tf-idf作为word2vec的词向量权重来
def cal_vector_using_tf_idf_word2vec(model, tuple_list):
    final_vec = []
    embed_dim = len(model["hello"])
    for t_l in tuple_list:
        s_vec = np.array([0.0] * embed_dim)
        for t in t_l:
            word = t[0]
            weight = t[1]
            # 词汇表中存在则进行处理,不存在就算了先
            if word in model.index2word:
                s_vec += model[word] * weight
        final_vec.append(s_vec)
    return final_vec


# 总方法，从word2vec中拿
def cal_feature_vec_from_word2vec(description_divide_file, pre_train_model_file=r"E:\研究生\word2vec\GoogleNews_vec.bin"):
    word_divide = dr.read_api_divide_description(description_divide_file)
    word_dic, weight = cal_tf_idf(word_divide)
    model = read_pre_trained_word2vec(pre_train_model_file)
    w_tf_tuple = convert_weight_into_tuple(word_dic, weight)
    final_vec = cal_vector_using_tf_idf_word2vec(model, w_tf_tuple)

    return final_vec


# 存储word2vec生成的向量
def save_word2vec_vec(description_file, save_file_name):
    features = cal_feature_vec_from_word2vec(description_file)
    print(len(features))
    features = np.array(features)
    print(len(features))
    features.tofile(save_file_name)
    print("{} 存储成功".format(save_file_name))


# 用bert——serving直接调bert服务
# server直接本地
# bert-serving-start -model_dir=E:bert/uncased_L-12_H-768_A-12/ -num_worker=1 -cpu -max_seq_len=150
def cal_feature_vec_from_bert_vec_mean(description_divide_file):
    word_divide = dr.read_api_divide_description(description_divide_file)
    print("开始计算bert预训练向量")
    bc = BertClient(check_length=False)
    f_v_list = []
    i = 0
    for w_d in word_divide:
        features_vec = bc.encode([w_d], is_tokenized=True)
        f_v_list.append(features_vec)
        print(i)
        i = i + 1
    print("计算成功")
    return f_v_list


# 保存bert结果
def save_bert_vec(description_file, save_file_name):
    b_f = cal_feature_vec_from_bert_vec_mean(description_file)
    n_b_f = np.array(b_f).squeeze(1)
    print(n_b_f.shape)
    n_b_f.tofile(save_file_name)
    print("{} 存储成功".format(save_file_name))


if __name__ == "__main__":
    t_a_d_file_name = "recommend_api_divide_description.txt"
    t_m_d_file_name = "recommend_mashup_divide_description.txt"
    # t_final_vec = cal_feature_vec_from_word2vec(t_a_d_file_name)
    # print(len(t_final_vec))
    # print(len(t_final_vec[0]))

    # t_tensor = torch.tensor(cal_feature_vec_from_word2vec(t_a_d_file_name))
    # print(t_tensor.size())
    t_save_w2v = "recommend_api_w2v_vec.txt"
    # save_word2vec_vec(t_a_d_file_name, t_save_w2v)

    t_save_bert = "recommend_api_bert_vec.txt"
    # save_bert_vec(t_a_d_file_name, t_save_bert)

    t_save_mashup_bert = "recommend_mashup_bert_vec.txt"
    save_bert_vec(t_m_d_file_name, t_save_mashup_bert)
