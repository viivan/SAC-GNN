"""
把各种设置信息全整到这里
"""
import argparse


# 各种设置
def init_parser():
    parser = argparse.ArgumentParser(description="args for MAIG")
    parser.add_argument('--cold_sim_num', type=int, default=5, help='sim num between test and train')

    parser.add_argument('--embed_num', type=int, default=64, help='embedding dimension of mashup and api')
    parser.add_argument('--layer_size', default=[64, 64, 64], help='output size of each gn_layer')
    parser.add_argument('--drop_out', default=[0.1, 0.1, 0.1], help='drop out for each gn_layer message')
    parser.add_argument('--graph_dropout', default=0.1, help='drop out rate of graph')
    parser.add_argument('--drop_out_fc', default=0.1, help='drop out of each fc in MLP')
    parser.add_argument('--mlp_layer_num', type=int, default=2, help='number of fc in MLP')
    parser.add_argument('--mlp_layer_dim', type=int, default=128, help='hidden size of MLP')
    parser.add_argument('--batch_size', type=int, default=200, help='batch size of train and test')
    parser.add_argument('--train_scale', default=0.9, help='scale of train data in whole')
    parser.add_argument('--device', default='cuda:0', help='my single gpu index')
    parser.add_argument('--epoch', type=int, default=10000, help='number of round')
    parser.add_argument('--learn_rate', default=0.001, help='learning rate')
    parser.add_argument('--regs', default=[1e-5], help='Regularization.')

    parser.add_argument('--loss_func', default='sample_bpr', help='loss function')
    parser.add_argument('--sample_num', type=int, default=1, help='number of pos and neg sample')
    parser.add_argument('--pos_sample_num', type=int, default=5, help='number of pos sample')
    parser.add_argument('--neg_sample_num', type=int, default=50, help='number of neg sample')

    parser.add_argument('--hr_k', type=int, default=10, help='k for hit ratio')
    parser.add_argument('--ndcg_k', type=int, default=10, help='k for NDCG')
    parser.add_argument('--map_k', type=int, default=10, help='k for mAP')

    parser.add_argument('--prepare_method', default='bert', help='ALCG input embeddings calculating method')
    parser.add_argument('--prepare_embed_dim_w2v', type=int, default=300,
                        help='ALCG input embeddings dimension for w2v')
    parser.add_argument('--prepare_embed_dim_bert', type=int, default=768,
                        help='ALCG input embeddings dimension for bert')

    parser.add_argument('--cpu_core', type=int, default=3, help='core count for parallel')
    return parser.parse_args()