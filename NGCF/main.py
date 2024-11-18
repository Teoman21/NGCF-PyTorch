'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

import torch
import torch.optim as optim
import torch.nn.functional as F
from NGCF import NGCF
from utility.helper import *
from utility.batch_test import *
import argparse
import warnings
from time import time

warnings.filterwarnings('ignore')

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='Dataset to use')
parser.add_argument('--regs', type=str, required=True, help='Regularization values')
parser.add_argument('--embed_size', type=int, default=64, help='Embedding size')
parser.add_argument('--layer_size', type=str, required=True, help='Layer sizes')
parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--save_flag', type=int, default=1, help='Save model flag')
parser.add_argument('--pretrain', type=int, default=0, help='Pretrain flag')
parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
parser.add_argument('--epoch', type=int, default=200, help='Number of epochs')
parser.add_argument('--verbose', type=int, default=50, help='Verbosity level')
parser.add_argument('--node_dropout', type=str, required=True, help='Node dropout rates')
parser.add_argument('--mess_dropout', type=str, required=True, help='Message dropout rates')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
parser.add_argument('--node_dropout_flag', type=bool, default=True, help='Enable or disable node dropout')
parser.add_argument('--weights_path', type=str, default='./weights/', help='Path to save model weights')


args = parser.parse_args()

# Automatically set the device to GPU if available, or fallback to CPU
if torch.cuda.is_available():
    args.device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    args.device = torch.device('cpu')
    print("CUDA not available. Using CPU.")

# Load adjacency matrices
plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

args.node_dropout = eval(args.node_dropout)
args.mess_dropout = eval(args.mess_dropout)

# Initialize NGCF model
model = NGCF(data_generator.n_users,
             data_generator.n_items,
             norm_adj,
             args).to(args.device)

t0 = time()
"""
*********************************************************
Train.
"""
cur_best_pre_0, stopping_step = 0, 0
optimizer = optim.Adam(model.parameters(), lr=args.lr)

loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
for epoch in range(args.epoch):
    t1 = time()
    loss, mf_loss, emb_loss = 0., 0., 0.
    n_batch = data_generator.n_train // args.batch_size + 1
    print('hello')

    for idx in range(n_batch):
        users, pos_items, neg_items = data_generator.sample()
        u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                       pos_items,
                                                                       neg_items,
                                                                       drop_flag=args.node_dropout_flag)

        batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                          pos_i_g_embeddings,
                                                                          neg_i_g_embeddings)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        

        loss += batch_loss
        mf_loss += batch_mf_loss
        emb_loss += batch_emb_loss

    if (epoch + 1) % 10 != 0:
        if args.verbose > 0 and epoch % args.verbose == 0:
            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                epoch, time() - t1, loss, mf_loss, emb_loss)
            print(perf_str)
        continue

    t2 = time()
    users_to_test = list(data_generator.test_set.keys())
    ret = test(model, users_to_test, drop_flag=False)

    t3 = time()

    loss_loger.append(loss)
    rec_loger.append(ret['recall'])
    pre_loger.append(ret['precision'])
    ndcg_loger.append(ret['ndcg'])
    hit_loger.append(ret['hit_ratio'])

    if args.verbose > 0:
        perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                   'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                   (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
                    ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                    ret['ndcg'][0], ret['ndcg'][-1])
        print(perf_str)

    cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                stopping_step, expected_order='acc', flag_step=5)

    # *********************************************************
    # Early stopping when cur_best_pre_0 is decreasing for ten successive steps.
    if should_stop:
        break

    # *********************************************************
    # Save the user & item embeddings for pretraining.
    if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
        torch.save(model.state_dict(), args.weights_path + str(epoch) + '.pkl')
        print('save the weights in path: ', args.weights_path + str(epoch) + '.pkl')

recs = np.array(rec_loger)
pres = np.array(pre_loger)
ndcgs = np.array(ndcg_loger)
hit = np.array(hit_loger)

best_rec_0 = max(recs[:, 0])
idx = list(recs[:, 0]).index(best_rec_0)

final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
             (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
              '\t'.join(['%.5f' % r for r in pres[idx]]),
              '\t'.join(['%.5f' % r for r in hit[idx]]),
              '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
print(final_perf)
