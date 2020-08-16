from __future__ import print_function, absolute_import, division
import argparse
import tensorflow as tf
import os
from models.deepwalk import Deepwalk
from load_data import *
from models.minibatch import EdgeMinibatchIterator
import random
import time
import numpy as np
from samplers.sampler import negative_sampling
from samplers.dfs import *
from sklearn.metrics import f1_score, roc_auc_score
from evaluate import link_predict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./data/arxiv/',
                        help='input dataset path.')
    parser.add_argument('--learning_rate', type=float, default=0.8,
                        help='learning rate.')
    parser.add_argument('--model_size', type=str, default="small",
                        help='big or small.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs.')
    parser.add_argument('--validate_iter', type=int, default=50,
                        help='how often to run a validation minibatch.')
    parser.add_argument('--print_step', type=int, default=50,
                        help='how often to print training info.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='number of batch size.')
    parser.add_argument('--validate_batch_size', type=int, default=64,
                        help='how many nodes per validation sample.')                 
    parser.add_argument('--dim', type=int, default=256,
                        help='size of output dim.')
    parser.add_argument('--max_degree', type=int, default=300,
                        help='maximum node degree.')
    parser.add_argument('--node_num', type=int, default=5242,
                        help='number of negative items for each pair.')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight for l2 loss on embedding matrix.')
    parser.add_argument('--identity_dim', type=int, default=50,
                        help='set to positive value to use identity embedding features of that dimension.')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout rate.')
    parser.add_argument('--walks_num', type=int, default=100,
                        help='length of walk per user node.')
    parser.add_argument('--save_size', type=int, default=1024,
                        help='number of saved_model batch size.')  
    parser.add_argument('--patience', type=int, default=50,
                        help='early stopping.')
    parser.add_argument('--out_dir', type=str, default="./embeddings/",
                        help='save embeddings path.')
    return parser.parse_args()

def save_embeddings(sess, model, minibatch_iter, size, out_dir, mod=""):
    embeddings = []
    finished = False
    seen1 = set([])
    nodes1 = []
    iter_num = 0
    name1 = 'embedding'

    while not finished:
        feed_dict, finished, edges = minibatch_iter.incremental_embed_feed_dict(size, iter_num)
        iter_num += 1
        outs = sess.run([model.outputs1, model.outputs2], feed_dict=feed_dict)
        for i, edge in enumerate(edges):
            if not edge[0] in seen1:
                embeddings.append(outs[0][i, :])
                nodes1.append(edge[0])
                seen1.add(edge[0])
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    embeddings = np.vstack(embeddings)
    np.save(out_dir + name1 + mod + ".npy", embeddings)
    with open(out_dir + name1 + mod + ".txt", "w") as fp:
        fp.write("\n".join(map(str, nodes1)))


def get_score(saved_model, node1, node2):
    try:
        embed_1 = saved_model['embeds'][int(node1)]
        embed_2 = saved_model['embeds'][int(node2)]
        result = np.dot(embed_1, embed_2)
        return result
    except Exception as e:
        pass

# link prediction
def evaluate(model, true_edges, false_edges):
    label_list = list()
    predict_list = list()
    num = len(true_edges)
    for edge in true_edges:
        score = get_score(model, edge[0], edge[1])
        predict_list.append(score)
        label_list.append(1)
    for edge in false_edges:
        score = get_score(model, edge[0], edge[1])
        predict_list.append(score)
        label_list.append(0)
    
    # set threshold
    score_sorted = predict_list[:]
    score_sorted.sort()
    threshold = score_sorted[-num]

    y_pred = np.zeros(len(predict_list), dtype=np.int32)
    for i in range(len(predict_list)):
        if predict_list[i] >= threshold:
            y_pred[i] = 1
    y_true = np.array(label_list)
    y_scores = np.array(predict_list)

    return np.mean(roc_auc_score(y_true, y_scores)), np.mean(f1_score(y_true, y_pred))


def train(G, train_data, args):
    # read data  
    features = None
    node_num = args.node_num  
    id_map = construct_id_map(node_num)
    vocab_size = len(id_map.values())

    # print("id_map", id_map)
    placeholders = {
        'batch1': tf.placeholder(tf.int32, shape=(None), name='batch1'),
        'batch2': tf.placeholder(tf.int32, shape=(None), name='batch2'),
        'batch3': tf.placeholder(tf.int32, shape=(None), name='batch3'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size': tf.placeholder(tf.int32, name='batch_size'),
    }

    context_pairs = load_walks(train_data)   

    minibatch = EdgeMinibatchIterator(G, 
            id_map,
            placeholders, batch_size=args.batch_size,
            max_degree=args.max_degree, 
            context_pairs = context_pairs)

    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")
    
    print("build model....")
    model = Deepwalk(placeholders,
                    vocab_size,
                    embedding_dim=args.dim,
                    lr=args.learning_rate)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session(config=config)
    # tensorboard
    summary_writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(init_op, feed_dict={adj_info_ph: minibatch.adj})
    train_adj_info = tf.assign(adj_info, minibatch.adj)
    
    # training
    q_1_dict, mask = load_item_pop(train_data)
    
    # DFS for each node to generate markov chain
    print("generating markov chain by DFS......")
    tt = time.time()
    candidates = candidate_choose(G, mask, args)
    print("time for generating candidates", time.time() - tt)

    N_steps = 10
    N_negs = 1
    best_auc = 0

    for epoch in range(args.epochs):
        print("epoch %d" % epoch)
        data_loader = Data_loader(args.batch_size)
        data_loader.load_train_data(train_data)
        data_loader.reset_pointer()
        for it in range(data_loader.num_batch):
            batch_data = data_loader.next_batch()
            node1 = [x[0] for x in batch_data]
            node2 = [x[1] for x in batch_data]

            # generate negative examples            
            if it == 0:
                start_given = None
            else:
                start_given = neg_examples   
            neg_examples = negative_sampling(model, sess, candidates, start_given, q_1_dict, N_steps, N_negs, node1, node2, args)   

            # update model params
            feed_dict={model.inputs1:node1, model.inputs2:node2, model.neg_samples:neg_examples, model.batch_size:args.batch_size}
            outs = sess.run([model.merged_loss, model.loss, model.opt_op, model.outputs1, model.outputs2, model.neg_outputs], feed_dict=feed_dict) 

            # add_summary for tensorboard show
            if it % args.print_step == 0:
                summary_writer.add_summary(outs[0], epoch *  data_loader.num_batch + it)

            if it % args.print_step == 0:
                print("model model", "Iter:", '%04d' % it, "d_loss=", "{:.5f}".format(outs[1]))

        # save model (node embeddings)
        saved_model = dict()
        embeds = list()  
        start_index = 0 
        while start_index < args.node_num:
            end_index = start_index + args.save_size
            if end_index < args.node_num:
                feed_dict = minibatch.feed_dict_val(start_index, end_index)
                embed = sess.run(model.outputs1, feed_dict=feed_dict)
            else:
                feed_dict = minibatch.feed_dict_val(start_index, node_num)
                embed = sess.run(model.outputs1, feed_dict=feed_dict)
            if start_index == 0:
                embeds = embed
            else:
                embeds = np.vstack((embeds, embed)) 
            start_index = end_index
        saved_model['embeds'] = np.array(embeds)

        # validation for each epoch with link prediction
        valid_auc_score, valid_f1_score = evaluate(saved_model, valid_data_true, valid_data_false)
        print("Epoch:", '%04d' % (epoch + 1),
                    "val_auc=", "{:.5f}".format(valid_auc_score),
                    "val_f1=", "{:.5f}".format(valid_f1_score))
        curr_auc = valid_auc_score
        if curr_auc > best_auc:
            best_auc = curr_auc
            patience = 0
        else:
            patience += 1
            if patience > args.patience:
                print("Early Stopping...")
                break
    # save model embeddings for downstream task
    save_embeddings(sess, model, minibatch, args.validate_batch_size, args.out_dir)
    print("training complete......") 

    # test for link prediction 
    test_auc_score, test_f1_score = link_predict(test_data_true, test_data_false, args)
    print("test_auc=", "{:.5f}".format(test_auc_score),
          "test_f1=","{:.5f}".format(test_f1_score))
   
if __name__ == "__main__":
    args = parse_args()
    filepath = args.input
    # load train_data, valid data
    train_data = load_edges(filepath + 'train.txt')
    valid_data_true, valid_data_false = load_test_data(filepath + 'valid.txt')
    test_data_true, test_data_false = load_test_data(filepath + 'test.txt')   
    G = construct_graph(train_data)
    train(G, train_data, args) 

