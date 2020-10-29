import numpy as np
from scipy import sparse
from numpy import array
import time
from collections import defaultdict
import copy
import random
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy.sparse import csr_matrix
from scipy import sparse

class Data_Loader():
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def load_train_data(self, positive_examples, args):
        train_data = defaultdict(list)
        for edge in positive_examples:
            train_data[edge[0]].append(edge[1])
        self.data = []
        user_list = list(train_data.keys())
        while len(self.data) < len(positive_examples):
            np.random.shuffle(user_list)
            for user_id in user_list:
                user_item_pairs = User_Seq_epoch(self.G, user_id, train_data, args)
                if len(user_item_pairs) == 0:
                    break
                self.data.extend(user_item_pairs)
        self.data = np.array(self.data)
        self.num_batch = int(self.data.shape[0] / self.batch_size)
        start_index = self.num_batch * self.batch_size
        end_index = start_index + self.batch_size
        if start_index < len(positive_examples):
            if end_index <= len(positive_examples):
                pass
            else:
                diff = end_index - len(positive_examples)
                self.add_data = self.data[0:diff]
                self.data = np.vstack((self.data, self.add_data))
                self.num_batch = self.num_batch + 1
        else:
            self.data = self.data[:start_index]
        self.data_D = np.split(self.data, self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):

        ret = self.data_D[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0
    

def construct_id_map(nodes_all):
    d = dict()
    for id in list(range(nodes_all)):
        d[int(id)] = int(id)
    return d


def construct_node(filename):
    nodes = np.loadtxt(filename)
    n = [int(n) for n in nodes]
    return n


def load_walks(train_edges):
    conversion = lambda n: int(n)
    walks = []
    for i in train_edges:
        line = [i[0], i[1]]
        #print("line", line)
        walks.append(map(conversion, line))
    return walks


def load_edges(filename):
    edges = list()
    with open(filename, 'r') as f:
        for line in f:
            user, item = line.strip().split('\t')
            edges.append((int(user), int(item))) 
    return edges


def str_list_to_int(str_list):
    return [int(item) for item in str_list]


def read_edges_from_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        edges = [str_list_to_int(line.strip().split('\t')) for line in lines]
    return edges


def read_nodes_from_file(filename):
    d = set()
    with open(filename, 'r') as f:
        for line in f:
            userid, itemid = str_list_to_int(line.strip().split('\t'))
            d.add(userid)
            d.add(itemid)
    a = list(d)
    return a

def load_neg(filename):
    ll=list()
    with open(filename, 'r') as f:
        for line in f:
            d=line.strip().split('\t')
            neg_list=[int(i) for i in d[1:]]
            ll.append(neg_list)
    return ll

from collections import defaultdict
def load_item_pop(X_train):
    item_pop = list()
    node_deg = dict()
    dd = defaultdict(list)
    for edge in X_train:
        dd[int(edge[1])].append(int(edge[0]))
    for key in dd.keys():
        item_pop.append(1)
    deg_sum = np.sum(item_pop)
    for key in dd.keys():
        node_deg[key] = 1/deg_sum
    return node_deg, dd

def construct_graph(edges):
    print("Construct the graph for training")
    G = nx.Graph()
    for edge in edges:
        x = edge[0]
        y = edge[1]
        G.add_edge(x, y)
    return G

def load_test_embedding(filename, embeddings):
    d = defaultdict(list)
    with open(filename, 'r') as f:
        num = 0
        for line in f:
            nodeid = line.strip()
            d[int(nodeid)].append(embeddings[num])
            num += 1
       
        a = sorted(d.items(), key=lambda x: x[0])
  
        a = np.array([y[0] for x, y in a])
        return a

def load_dict(edges):
    d = defaultdict(list)
    for edge in edges:
        user = int(edge[0])
        item = int(edge[1])
        d[user].append(item)
    return d

def load_test_neg(train_data, valid_data, test_data, args):
    ll = list()
    all_edges = np.vstack((train_data, valid_data, test_data))
    d_dict = load_dict(all_edges)
    for edge in test_data:
        user = int(edge[0])
        item = int(edge[1])
        items_list = list(range(args.user_num, args.user_num+args.item_num))
        for id in d_dict[user]:
            if id in items_list:
                items_list.remove(id)
        if args.input == './data/ml-100k/':
            samples = items_list
        else:
            samples = np.random.choice(items_list, 500)
        ll.append(samples)
    return ll


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(nodes):
    """Row-normalize feature matrix and convert to tuple representation"""
    node = list(range(nodes))
    row = np.array([i for i in node])
    col = np.array([i for i in node])
    data = np.array([1.0 for i in node])
    a = csr_matrix((data, (row,col)), shape=(len(node), len(node)), dtype=np.float32).toarray()
    features = sparse.csr_matrix(a) 
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)



def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def load_adj(train_data, node_num):
    dd = defaultdict(set)
    graph = defaultdict(list)
    for edge in train_data:
        node1 = edge[0]
        node2 = edge[1]
        dd[node1].add(node2)
        dd[node2].add(node1)
    a = sorted(dd.items(), key=lambda x: int(x[0]))
    iter = 0
    for x in a:
        if iter == int(x[0]):
            for id in x[1]:
                graph[int(x[0])].append(int(id))
        else:
            flag = int(x[0]) - iter
            for j in list(range(flag)):
                graph[j + iter].append(j+iter)
            for i in x[1]:
                graph[int(x[0])].append(int(i))
            iter = iter + flag
        iter += 1
    if iter < node_num:
        for i in list(range(node_num - iter)):
            graph[iter].append(iter)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    return adj
