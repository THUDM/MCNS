from __future__ import division
from __future__ import print_function

import numpy as np
from load_data import *
import time
np.random.seed(123)


class EdgeMinibatchIterator(object):
    """ This minibatch iterator iterates over batches of sampled edges or
    random pairs of co-occuring edges.

    G -- networkx graph
    id2idx -- dict mapping node ids to index in feature tensor
    placeholders -- tensorflow placeholders object
    context_pairs -- if not none, then a list of co-occuring node pairs (from random walks)
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    n2v_retrain -- signals that the iterator is being used to add new embeddings to a n2v model
    fixed_n2v -- signals that the iterator is being used to retrain n2v with only existing nodes as context
    """

    def __init__(self, G, id2idx,
                 placeholders, context_pairs=None, batch_size=100, max_degree=25,
                 **kwargs):
        self.G = G
        self.id2idx = id2idx
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0
        self.nodes = G.nodes()
        
        self.save_embed_nodes = list(self.id2idx.values())
        self.save_embed_nodes = np.random.permutation(self.save_embed_nodes)
        
        self.nodes = np.random.permutation(G.nodes())
        edges = context_pairs
        self.train_edges = self.edges = np.random.permutation(edges)
        self.train_edges = self._remove_isolated(self.train_edges)

        self.adj, self.deg = self.construct_adj()

    def construct_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        deg = np.zeros((len(self.id2idx),))

        for nodeid in self.G.nodes():
            neighbors = np.array([self.id2idx[neighbor] 
                for neighbor in self.G.neighbors(nodeid)])
            deg[self.id2idx[nodeid]] = len(neighbors)
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj, deg
        
    def _remove_isolated(self, edge_list):
        new_edge_list = []
        for n1, n2 in edge_list:
            new_edge_list.append((n1, n2))
        return new_edge_list

    def batch_feed_dict_val(self, batch_edges):
        batch1 = []
        batch2 = []
        # batch3 = []
        for node1, node2 in batch_edges:
            batch1.append(self.id2idx[node1])
            batch2.append(self.id2idx[node2])

        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size']: len(batch_edges)})
        feed_dict.update({self.placeholders['batch1']: batch1})
        feed_dict.update({self.placeholders['batch2']: batch2})
        # feed_dict.update({self.placeholders['batch3']: batch3})
        return feed_dict, batch1, batch2

    def batch_feed_dict(self, batch_edges):
        batch1 = []
        batch2 = []
        batch3 = []
        for node1, node2 in batch_edges:
            batch1.append(self.id2idx[node1])
            batch2.append(self.id2idx[node2])

        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size']: len(batch_edges)})
        feed_dict.update({self.placeholders['batch1']: batch1})
        feed_dict.update({self.placeholders['batch2']: batch2})
        feed_dict.update({self.placeholders['batch3']: batch3})
        return feed_dict
    
    def batch_feed_dict_embeddings(self, batch_edges):
        batch1 = []
        batch2 = []
        for node1, node2 in batch_edges:
            batch1.append(self.id2idx[node1])
            batch2.append(self.id2idx[node2])
        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size']:len(batch_edges)})
        feed_dict.update({self.placeholders['batch1']:batch1})
        feed_dict.update({self.placeholders['batch2']:batch2})
        return feed_dict


    def val_feed_dict(self, val_edges, size=None):
        edge_list = val_edges
        if size is None:
            return self.batch_feed_dict(edge_list)
        else:
            ind = np.random.permutation(len(edge_list))
            val_edges = [edge_list[i] for i in ind[:min(size, len(ind))]]
        return self.batch_feed_dict_val(val_edges)

    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = self.save_embed_nodes
        val_nodes = node_list[iter_num * size:min((iter_num + 1) * size,
                                                  len(node_list))]
        val_edges = [(n, n) for n in val_nodes]
        return self.batch_feed_dict_embeddings(val_edges), (iter_num + 1) * size >= len(node_list), val_edges
