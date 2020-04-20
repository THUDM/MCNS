from __future__ import print_function
import numpy as np
from load_data import *

def top_k(scores, target, hit30):
    scores = scores.reshape(-1)
    idx = np.argsort(-scores)
    find_target = idx == target
    ranks = np.nonzero(find_target)
    target_rank = ranks[0][0] + 1
    mrr = np.divide(1.0, target_rank)
    hit30= int(target_rank<=30)
    return mrr, hit30

def metric(test_edges, negatives, user_embeds, item_embeds, args=None):
    mrr_tot = 0
    hit30_tot = 0
    count = 0 
    for i in range(len(test_edges)):
        u_id = test_edges[i][0]
        v_id = test_edges[i][1] - args.user_num
        aff = np.dot(user_embeds[u_id], item_embeds[v_id]) 
        items = np.array([item_embeds[x - args.user_num] for x in negatives[i]]) 
        aff_neg = np.matmul(user_embeds[u_id], items.T)
        aff_all = np.append(aff_neg, aff) 
        target = len(negatives[i])
        mrr, hit30 = top_k(aff_all, target, 30)
        mrr_tot += mrr 
        hit30_tot += hit30
        count += 1
    MRR = mrr_tot / count
    HITS_30 = hit30_tot /count
    return MRR, HITS_30

def recommend(train_data, valid_data, test_data, args):
    print("loading embedding...")
    embeds_u = np.load(args.save_dir + "/embedding_u.npy")
    test_embeds_u = load_test_embedding(args.save_dir + "/embedding_u.txt", embeds_u)
    print('test_embeds', test_embeds_u.shape)

    embeds_v = np.load(args.save_dir + "/embedding_v.npy")
    test_embeds_v = load_test_embedding(args.save_dir + "/embedding_v.txt", embeds_v)
    print('test_embeds', test_embeds_v.shape)

    user_embeds = test_embeds_u[:args.user_num]
    item_embeds = test_embeds_v[args.user_num:]
    print('item_embeds', item_embeds.shape)
    print('user_embeds', user_embeds.shape)

    negatives = load_test_neg(train_data, valid_data, test_data, args)
    # negatives = load_neg(args.input + 'test_neg.txt')
    print("Running recommendation...") 
    mrr, hit30 = metric(test_data, negatives, user_embeds, item_embeds, args)
    return mrr, hit30


