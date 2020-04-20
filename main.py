from __future__ import print_function, absolute_import, division
import tensorflow as tf
import argparse
import os
from models.graphsage import Graphsage, UniformNeighborSampler, SAGEInfo
from load_data import *
from models.minibatch import EdgeMinibatchIterator
import random
import time
import numpy as np
import math
from samplers.sampler import negative_sampling
from samplers.dfs import *
from evaluate import recommend

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./data/amazon/',
                        help='input dataset path.')
    parser.add_argument('--model', type=str, default='graphsage_mean',
                        help='model name.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate.')
    parser.add_argument('--model_size', type=str, default="small",
                        help='big or small.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs.')
    parser.add_argument('--validate_iter', type=int, default=50,
                        help='how often to run a validation minibatch.')
    parser.add_argument('--print_step', type=int, default=50,
                        help='how often to print training info.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='number of batch size.')
    parser.add_argument('--validate_batch_size', type=int, default=512,
                        help='how many nodes per validation sample.')                
    parser.add_argument('--samples_1', type=int, default=25,
                        help='number of samples in  layer 1.')
    parser.add_argument('--samples_2', type=int, default=10,
                        help='number of samples in layer 2.')
    parser.add_argument('--dim_1', type=int, default=256,
                        help='size of output dim.')
    parser.add_argument('--dim_2', type=int, default=256,
                        help='size of output dim.')
    parser.add_argument('--max_degree', type=int, default=300,
                        help='maximum node degree.')
    parser.add_argument('--user_num', type=int, default=0,
                        help='number of negative items for each pair.')
    parser.add_argument('--item_num', type=int, default=0,
                        help='number of negative items for each pair.')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight for l2 loss on embedding matrix.')
    parser.add_argument('--identity_dim', type=int, default=50,
                        help='set to positive value to use identity embedding features of that dimension.')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout rate.')
    parser.add_argument('--walks_num', type=int, default=100,
                        help='length of walk per user node.')
    parser.add_argument('--patience', type=int, default=10,
                        help='early stopping.')
    parser.add_argument('--save_dir', type=str, default="./embeddings/",
                        help='save embeddings path.')
    return parser.parse_args()


def save_embeddings(sess, model, minibatch_iter, size, out_dir, mod=""):
    embeddings_u = []
    embeddings_v = []
    finished = False
    seen1 = set([])
    nodes1 = []
    seen2 = set([])
    nodes2 = []
    iter_num = 0
    name1 = 'embedding_u'
    name2 = 'embedding_v'

    while not finished:
        feed_dict, finished, edges = minibatch_iter.incremental_embed_feed_dict(size, iter_num)
        iter_num += 1
        outs = sess.run([model.outputs1, model.outputs2], feed_dict=feed_dict)
        for i, edge in enumerate(edges):
            if not edge[0] in seen1:
                embeddings_u.append(outs[0][i, :])
                nodes1.append(edge[0])
                seen1.add(edge[0])
        for i, edge in enumerate(edges):
            if not edge[0] in seen2:
                embeddings_v.append(outs[1][i, :])
                nodes2.append(edge[0])
                seen2.add(edge[0])
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    embeddings_u = np.vstack(embeddings_u)
    np.save(out_dir + name1 + mod + ".npy", embeddings_u)
    with open(out_dir + name1 + mod + ".txt", "w") as fp:
        fp.write("\n".join(map(str, nodes1)))

    embeddings_v = np.vstack(embeddings_v)
    np.save(out_dir + name2 + mod + ".npy", embeddings_v)
    with open(out_dir + name2 + mod + ".txt", "w") as fp:
        fp.write("\n".join(map(str, nodes2)))


def evaluate(sess, model, placeholders, minibatch_iter, candidates, q_1_dict, N_steps, N_negs, valid_data, size=None):
    t_test = time.time()
    feed_dict_val, node1, node2 = minibatch_iter.val_feed_dict(valid_data, size)
    start_given = None
    neg_examples = negative_sampling(model, sess, candidates, start_given, q_1_dict, N_steps, N_negs, node1, node2, args)
    feed_dict_val.update({placeholders['batch3']: neg_examples})
    feed_dict_val.update({placeholders['batch4']: size})
    outs_val = sess.run([model.loss, model.ranks, model.mrr],
                    feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)


def train(G, train_data, valid_data, test_data, args):
    # read data 
    features = None 
    nodes_all = args.user_num + args.item_num
    id_map = construct_id_map(nodes_all)
    placeholders = {
        'batch1': tf.placeholder(tf.int32, shape=(None), name='batch1'),
        'batch2': tf.placeholder(tf.int32, shape=(None), name='batch2'),
        'batch3': tf.placeholder(tf.int32, shape=(None), name='batch3'),
        'batch4': tf.placeholder(tf.int32, shape=(None), name='batch4'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size': tf.placeholder(tf.int32, name='batch_size'),
    }
    context_pairs = load_walks(train_data)
    minibatch = EdgeMinibatchIterator(G,
                                      id_map,
                                      placeholders, batch_size=args.batch_size,
                                      max_degree=args.max_degree,
                                      context_pairs=context_pairs)
    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")
    
    print("build model....")
    # neighbors sampling
    sampler = UniformNeighborSampler(adj_info)
    # two layers sampling information
    layer_infos = [SAGEInfo("node", sampler, args.samples_1, args.dim_1),
                   SAGEInfo("node", sampler, args.samples_2, args.dim_2)]
    model = Graphsage(placeholders,
                         features,
                         adj_info,
                         args.learning_rate,
                         layer_infos=layer_infos,
                         model_size=args.model_size,
                         identity_dim=args.identity_dim,
                         args=args,
                         logging=True) 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session(config=config)
    # tensorboard
    summary_writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(init_op, feed_dict={adj_info_ph: minibatch.adj})
    train_adj_info = tf.assign(adj_info, minibatch.adj)
    
    # training
    t1 = time.time()
    q_1_dict, mask = load_item_pop(train_data)
    
    # DFS for each node to generate markov chain
    print("generating markov chain by DFS......")
    tt = time.time()
    candidates = candidate_choose(G, mask, args)
    print("time for generating negative examples", time.time() - tt)
    print("candidates", candidates)

    N_steps = 10
    N_negs = 1
    best_mrr = 0 

    for epoch in range(args.epochs):
        print("epoch %d" % epoch)
        data_loader = Data_Loader(args.batch_size)
        data_loader.load_train_data(train_data)
        data_loader.reset_pointer()
        for it in range(data_loader.num_batch):
            batch_data = data_loader.next_batch()
            node1 = [x[0] for x in batch_data]
            node2 = [x[1] for x in batch_data]

            # generate negative examples with MCNS
            t0 = time.time()
            if it == 0:
                start_given = None
            else:
                start_given = generate_examples
            generate_examples = negative_sampling(model, sess, candidates, start_given, q_1_dict, N_steps, N_negs, node1, node2, args)
            print("time for generating negative examples", time.time() - t0)   

            # update model params
            feed_dict={model.inputs1:node1, model.inputs2:node2, model.neg_samples:generate_examples, model.batch_size:args.batch_size, model.number: args.batch_size}
            outs = sess.run([model.merged_loss, model.merged_mrr, model.loss, model.mrr, model.opt_op, model.outputs1, model.outputs2, model.neg_outputs], feed_dict=feed_dict) 
           
            # add_summary for tensorboard show
            if it % args.print_step == 0:
                summary_writer.add_summary(outs[0], epoch *  data_loader.num_batch + it)
                summary_writer.add_summary(outs[1], epoch *  data_loader.num_batch + it)
            if it % args.validate_iter == 0:
                t2 = time.time()
                val_cost, ranks, val_mrr, duration = evaluate(sess, model, placeholders, minibatch, candidates, q_1_dict, N_steps, N_negs, valid_data, size=args.validate_batch_size)
                print("evaluate time", time.time() - t2)
            if it % args.print_step == 0:
                print("model model", "Iter:", '%04d' % it, "d_loss=", "{:.5f}".format(outs[2]), "d_mrr=","{:.5f}".format(outs[3]))
                print("validation model", "Iter:", '%04d' % it, "val_loss=", "{:.5f}".format(val_cost), "val_mrr=","{:.5f}".format(val_mrr))
        
        # validation for early stopping......
        val_cost, ranks, val_mrr, duration = evaluate(sess, model, placeholders, minibatch, candidates, q_1_dict, N_steps, N_negs, valid_data, size=args.validate_batch_size)
        curr_mrr = val_mrr
        if curr_mrr > best_mrr:
            best_mrr = curr_mrr
            patience = 0
        else:
            patience += 1
            if patience > args.patience:
                print("Early Stopping...")
                break

    # save model embeddings for downstream task
    save_embeddings(sess, model, minibatch, args.validate_batch_size, args.save_dir)
    print("training complete......") 

    # test for recommendation......
    # mrr, hit30 = recommend(test_data, args) 
    mrr, hit30 = recommend(train_data, valid_data, test_data, args) 
    print("test_mrr=", "{:.5f}".format(mrr),
          "test_hit30=", "{:.5f}".format(hit30))


if __name__ == "__main__":
    args = parse_args()
    filepath = args.input
    # load train_data, valid data
    train_data = load_edges(filepath + 'train.txt')
    print("train_data", len(train_data))
    valid_data = load_edges(filepath + 'valid.txt')
    print("valid_data", len(valid_data))    
    test_data = read_edges_from_file(filepath + 'test.txt')
    G = construct_graph(train_data)
    train(G, train_data, valid_data, test_data, args)

    
