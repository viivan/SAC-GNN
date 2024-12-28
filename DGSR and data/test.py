import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import MAIGNet
import model_test
import time
import heapq
import metric
import multiprocessing
from functools import partial
import numpy as np
import scipy.sparse as sp

if __name__ == "__main__":
    n_dim = 10
    e1 = torch.rand([3, 5, 10], out=None)
    t1 = torch.rand([5], out=None)
    t2 = torch.rand([5], out=None)
    t3 = torch.rand([5], out=None)
    """
    up = torch.mm(t1, t2.t())  # [E,dim]*[dim,E]
    up = torch.divide(up, math.sqrt(3))
    output = torch.sum(up, dim=1)  # [E,1]
    """

    t = [t1, t2, t3]
    t = torch.stack(t)
    a_s = F.softmax(t, dim=0)
    print(a_s)

    a_s = torch.unsqueeze(a_s, 2)
    print(a_s.size())
    print(a_s)
    a_s = torch.repeat_interleave(a_s, repeats=n_dim, dim=2)
    print(a_s.size())

    print(e1)
    r = torch.mul(e1, a_s)  # 3,5,10
    print(r)
    print(r.size())

    # 对应tensor连接
    result = []
    for i in range(r.size()[1]):
        result.append(torch.cat([r[x][i] for x in range(r.size()[0])], dim=0))
    result = torch.stack(result)
    print(result)
    print(result.size())

    mlp = MAIGNet.MLP(30, 50, 10, 0, 0.1, batch_norm=True)
    output = mlp(result)
    print(output)
    print(output.size())

    m_num = 3
    m_i = [0, 2]
    a_i = [0, 1]
    a_i = [(x+m_num) for x in a_i]
    print(output[m_i])
    print(output[a_i])

    r1 = torch.rand([5, 10], out=None)
    r2 = torch.rand([5, 10], out=None)
    scores = torch.sum(torch.mul(r1, r2), dim=1)
    print(scores)
    maxi = nn.LogSigmoid()(scores + scores)
    mf_loss = -1 * torch.mean(maxi)
    print(mf_loss)

    g_in = nn.Linear(20, 1)
    r12 = torch.cat([r1, r2], dim=1)
    g = g_in(r12)
    print(g)
    g = nn.Sigmoid()(g)
    r_out = r1 * g + r2*(1 - g)
    print(r_out.size())

    test_mashup_index = [0, 1, 2]
    m_a_id_list = [[0, 1],
                   [0, 2],
                   [1, 2, 3]]
    recommend_index = [[2, 1, 3, 0],
                       [3, 1, 0, 2],
                       [1, 2, 3, 0]]
    metric.cal_average_hrk_metric(test_mashup_index, m_a_id_list, recommend_index, 3)
    metric.cal_average_ndcg_k(test_mashup_index, m_a_id_list, recommend_index, 10)

    test_mashup_index = [0]
    m_a_id_list = [[0, 2, 5]]
    recommend_index = [[0, 1, 2, 3, 4, 5]]
    metric.cal_map(test_mashup_index, m_a_id_list, recommend_index, 10)

    recommend_score = [[23,43,113,42],
                       [86,34,96,18],
                       [28,39,62,43]]
    result = model_test.sort_api_score_for_top(3, recommend_score[0])

    print("?")
    recommend_score = [[23,43,113,42],
                       [86,34,96,18],
                       [28,39,62,43]]
    pool = multiprocessing.Pool(2)
    func = partial(model_test.sort_api_score_for_top, 3)
    sorted_recommend_index = pool.map_async(func, recommend_score).get()
    pool.close()
    pool.join()
    print(sorted_recommend_index)
    print("akb48")

    final_vec = np.array([0.0] * 10)
    r = np.random.rand(3, 10)
    print(r)
    weight = [1.0, 2.0, 3.0]

    for i in range(3):
        final_vec += r[i] * weight[i]

    print(final_vec)

    n1 = 0.79745324
    n2 = 0.89093753
    n3 = 0.06652918
    print(n1 + 2 * n2 + 3 * n3)

    t = np.random.rand(3,1,3)
    print(t)
    t = t.squeeze(1)
    print(t)
    print(t.shape)

    t_m = [0, 1, 2]
    t_t = [[1, 2, 4, 3, 5], [2, 0, 1, 4, 5], [2, 3, 1, 7, 8]]
    print(sorted_recommend_index)
    metric.cal_average_hrk_metric(t_m, t_t, sorted_recommend_index, 3)
    metric.cal_average_ndcg_k(t_m, t_t, sorted_recommend_index, 3)

    t = torch.rand((2, 3))
    print(t)
    print(t[[0, 1, 0, 1]])

    print()
    t = np.random.random((2, 3))
    print(t)
    sp_t = sp.dok_matrix(t)
    print(sp_t)
    noise_shape = sp_t.nnz
    print(noise_shape)
    rate = 0.1
    random_tensor = 1 - rate
    tr = torch.rand(noise_shape)
    print(tr)
    random_tensor += tr
    print(random_tensor)
    dropout_mask = torch.floor(random_tensor).type(torch.bool)
    print(dropout_mask)
    dropout_mask[2] = False
    print(dropout_mask)

    sp_t = sp_t.tocoo()
    indices = torch.from_numpy(
        np.vstack((sp_t.row, sp_t.col)).astype(np.int64))
    values = torch.from_numpy(sp_t.data)
    ts_t = torch.sparse.FloatTensor(indices, values, sp_t.shape)
    i = ts_t._indices()
    v = ts_t._values()
    print(i)
    print(v)
    i = i[:, dropout_mask]
    v = v[dropout_mask]
    print(i)
    print(v)
    out = torch.sparse.FloatTensor(i, v, ts_t.shape)
    print(out)
    out = out * (1. / (1 - rate))
    print(out)

    def whole_bpr_loss(mashup_embeddings, pos_api_embeddings, neg_api_embeddings):
        #  正负项用来算bpr
        # 为了方便计算，每个mashup采样一个pos和neg来进行计算
        pos_scores = torch.sum(torch.mul(mashup_embeddings, pos_api_embeddings), dim=1)
        neg_scores = torch.sum(torch.mul(mashup_embeddings, neg_api_embeddings), dim=1)
        maxi = nn.LogSigmoid()(pos_scores - neg_scores)
        mf_loss = -1 * torch.mean(maxi)

        regular = (torch.norm(mashup_embeddings) ** 2
                       + torch.norm(pos_api_embeddings) ** 2
                       + torch.norm(neg_api_embeddings) ** 2) / 2
        print(torch.norm(mashup_embeddings))
        print(torch.norm(mashup_embeddings).shape)
        emb_loss = 0.1 * regular
        print(regular.shape)
        print(emb_loss.shape)
        return mf_loss + emb_loss

    t_m = torch.rand((3, 3))
    p_m = torch.rand((3, 3))
    n_m = torch.rand((3, 3))
    whole_bpr_loss(t_m, p_m, n_m)

    t = [3, 4, 1, 2, 5]
    b = sorted(t)
    g = [t.index(x) for x in b]
    print(b)
    print(g)
    print(t)

    t = torch.arange(12).view(4, 3)
    t = t.float()
    print(t)
    t_mean = torch.mean(t, dim=0)
    print(t_mean)
    print(t_mean.shape)
    print(t_mean.shape[0])
    t_mean = t_mean.view(1, 3)
    print(t_mean)
    print(t_mean.shape)
    print(t_mean.shape[0])

    t = [t_mean, t_mean, t_mean]
    print(t)
    t_whole = torch.cat(t, dim=0)
    print(t_whole)



