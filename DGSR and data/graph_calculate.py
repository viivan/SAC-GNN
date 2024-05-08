"""
构建图信息，邻接矩阵以及归一化邻接矩阵
MAIG(Mashup Api Invoke Graph)；ALCG(Api Label Connect Graph)
"""
import scipy.sparse as sp
import numpy as np
import data_read


# 计算归一化邻接矩阵（拉普拉斯）(无自连接)
# 输入为dok的矩阵数据
def cal_normalized_adjacency_matrix(adjacency_matrix):
    # D^-1/2 * A * D^-1/2
    row_sum = np.array(adjacency_matrix.sum(1))  # 顶点度

    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    bi_lap = d_mat_inv_sqrt.dot(adjacency_matrix).dot(d_mat_inv_sqrt)
    return bi_lap.tocoo()


# 构造MAIG稀疏邻接矩阵
# 返回为csr化的稀疏矩阵和归一化稀疏矩阵
def create_scipy_adjacency_matrix_MAIG(m_id, a_id, m_a_id):
    mashup_num = len(m_id)
    api_num = len(a_id)

    use_matrix = sp.dok_matrix((mashup_num, api_num), dtype=np.float)
    # 根据调用情况填空
    for i in range(len(m_a_id)):
        for a in m_a_id[i]:
            use_matrix[i, int(a)] = 1

    # 构建邻接矩阵
    adjacency_matrix = sp.dok_matrix((mashup_num + api_num, mashup_num + api_num), dtype=np.float)
    adjacency_matrix = adjacency_matrix.tolil()
    use_matrix = use_matrix.tolil()
    adjacency_matrix[:mashup_num, mashup_num:] = use_matrix
    adjacency_matrix[mashup_num:, :mashup_num] = use_matrix.T
    adjacency_matrix = adjacency_matrix.todok()
    print("完成mashup-api邻接矩阵创建")

    # 计算归一化邻接矩阵
    norm_adjacency_matrix = cal_normalized_adjacency_matrix(adjacency_matrix)
    print("完成mashup-api邻接矩阵归一化")
    return adjacency_matrix.tocsr(), norm_adjacency_matrix.tocsr()


# 构造ALCG稀疏邻接矩阵（无边类型）
# a_id为api序号，a_a_id为a_id对应的api相邻apis
def create_scipy_adjacency_matrix_ALCG_no_type(a_id, a_a_id):
    api_num = len(a_id)
    # 调用矩阵
    use_matrix = sp.dok_matrix((api_num, api_num), dtype=np.float)
    # 根据调用情况填空
    for i in range(len(a_a_id)):
        for a in a_a_id[i]:
            use_matrix[i, int(a)] = 1
    print("完成api邻接矩阵创建")

    # 计算归一化邻接矩阵
    norm_adjacency_matrix = cal_normalized_adjacency_matrix(use_matrix)
    print("完成api邻接矩阵归一化")
    return use_matrix.tocsr(), norm_adjacency_matrix.tocsr()


if __name__ == "__main__":
    """
    # mashup图构建
    t_MAIG_file_name = "recommend_mashup_api_id_more_than_3.txt"
    t_m_id, t_a_id, t_m_a_list = data_read.read_mashup_api(t_MAIG_file_name)
    MAIG_adj_matrix, MAIG_norm_adj_matrix = create_scipy_adjacency_matrix_MAIG(t_m_id, t_a_id, t_m_a_list)
    print(MAIG_adj_matrix)
    """

    # api图构建
    t_ALCG_file_name = "recommend_api_share_tag_id_more_than_3.txt"
    t_a_id, t_a_a_list = data_read.read_api(t_ALCG_file_name)
    ALCG_adj_matrix, ALCG_norm_adj_matrix = create_scipy_adjacency_matrix_ALCG_no_type(t_a_id, t_a_a_list)
    print(ALCG_adj_matrix)