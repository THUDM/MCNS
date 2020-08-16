import numpy as np
from scipy import sparse
from numpy import array

class Data_loader():
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def load_train_data(self, positive_examples):
        self.data = np.array(positive_examples)
        self.data = np.random.permutation(self.data)
        self.num_batch = int(len(positive_examples) / self.batch_size)
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
                self.data_D = np.split(self.data, self.num_batch, 0)
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


def construct_id_map(node_num):
    d = dict()
    for id in list(range(node_num)):
        d[int(id)] = int(id)
    return d


def load_walks(train_edges):
    conversion = lambda n: int(n)
    walks = []
    for i in train_edges:
        line = [i[0], i[1]]
        #print("line", line)
        walks.append(map(conversion, line))
    return walks


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

import networkx as nx
from collections import defaultdict
import numpy as np

def construct_graph(edges):
    print("Construct the graph for training")
    G = nx.Graph()
    for edge in edges:
        x = edge[0]
        y = edge[1]
        G.add_edge(x, y)
    return G

from collections import defaultdict
def load_item_pop(X_train):
    item_pop = list()
    node_deg = dict()
    dd = defaultdict(list)
    for edge in X_train:
        dd[int(edge[0])].append(int(edge[1]))
        dd[int(edge[1])].append(int(edge[0]))

    for key in dd.keys():
        item_pop.append(1)
    deg_sum = np.sum(item_pop)
    for key in dd.keys():
        node_deg[key] = 1 /deg_sum
    return node_deg, dd


def load_edges(filename):
    edges = list()
    with open(filename, 'r') as f:
        for line in f:
            user, item = line.strip().split('\t')
            edges.append((int(user), int(item))) 
    return edges

# load test data
def load_test_data(filename):
    print("Loading test/valid data......")
    true_edges = list()
    false_edges = list()
    with open(filename, 'r') as f:
        for line in f:
            user, item, label = line.strip().split('\t')
            if int(label) == 1:
                true_edges.append((int(user), int(item)))
            else:
                false_edges.append((int(user), int(item)))
    return true_edges, false_edges

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