import tensorflow as tf
from models.prediction import BipartiteEdgePredLayer
from models.layers import Layer, GraphConvolution
import numpy as np

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}  # all variable collections
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError


class GCN(Model):
    def __init__(self, placeholders, input_dim, embedding_dim=50, lr=0.001, args=None, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['feats']
        self.inputs1 = placeholders["batch1"]
        self.inputs2 = placeholders["batch2"]
        self.neg_samples = placeholders["batch3"]
        self.input_dim = input_dim  
        self.output_dim = embedding_dim  
        self.batch_size = placeholders['batch_size']
        self.number = placeholders["batch4"]
        self.placeholders = placeholders
        self.args = args
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += self.args.weight_decay * tf.nn.l2_loss(var)

        self.loss += self.link_pred_layer.loss(self.outputs1, self.outputs2, self.neg_outputs)
        self.loss = self.loss / tf.cast(self.batch_size, tf.float32)
        self.merged_loss = tf.summary.scalar('merged_loss', self.loss)

    def _accuracy(self):
        # shape: [batch_size]
        aff = self.link_pred_layer.affinity(self.outputs1, self.outputs2)
        neg_aff = self.link_pred_layer.affinity(self.outputs1, self.neg_outputs)
        self.neg_aff = tf.expand_dims(neg_aff, axis=1)
        _aff = tf.expand_dims(aff, axis=1)
        self.aff_all = tf.concat(axis=1, values=[self.neg_aff, _aff])
        size = tf.shape(self.aff_all)[1]
        _, indices_of_ranks = tf.nn.top_k(self.aff_all, k=size)
        _, self.ranks = tf.nn.top_k(-indices_of_ranks, k=size)
        self.mrr = tf.reduce_mean(tf.div(1.0, tf.cast(self.ranks[:, -1] + 1, tf.float32)))
        self.merged_mrr = tf.summary.scalar('merged_mrr', self.mrr)

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.embedding = self.activations[-1]  # embedding matrix
        self.embedding = tf.nn.l2_normalize(self.embedding, 1)

        self.outputs1 = tf.nn.embedding_lookup(self.embedding, self.inputs1)
        self.outputs2 = tf.nn.embedding_lookup(self.embedding, self.inputs2)
        self.neg_outputs =tf.nn.embedding_lookup(self.embedding, self.neg_samples)

        self.link_pred_layer = BipartiteEdgePredLayer(self.output_dim, self.output_dim, self.placeholders, act=tf.nn.sigmoid,
                                                      bilinear_weights=False,
                                                      name='edge_predict')

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()
        self.opt_op = self.optimizer.minimize(self.loss)
        self.p_probs = self.link_pred_layer.get_probs(self.outputs1, self.outputs2)

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.args.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.tanh,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=self.args.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=tf.nn.tanh,
                                            dropout=True,
                                            logging=self.logging))
        


 
