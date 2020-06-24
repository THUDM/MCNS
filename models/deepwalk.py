import tensorflow as tf
import math
from models.prediction import BipartiteEdgePredLayer

class Deepwalk(object):
    def __init__(self, placeholders, dict_size, name=None,
                 embedding_dim=50, lr=0.001, **kwargs):
        """ Simple version of Node2Vec/DeepWalk algorithm.

        Args:
            dict_size: the total number of nodes.
            degrees: numpy array of node degrees, ordered as in the data's id_map
            nodevec_dim: dimension of the vector representation of node.
            lr: learning rate of optimizer.
        """

        super(Deepwalk, self).__init__(**kwargs)

        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

        self.margin = 0.1

        self.placeholders = placeholders
        self.dict_size = dict_size
        self.embedding_dim = embedding_dim
        self.inputs1 = placeholders["batch1"]
        self.inputs2 = placeholders["batch2"]
        self.neg_samples = placeholders["batch3"]
        self.number = placeholders["batch4"]
        self.batch_size = placeholders['batch_size']
        

        # Model parameters
        self.loss = 0
        self.accuracy = 0
        
        # tensorflow word2vec tutorial
        self.target_embeds = tf.Variable(
            tf.random_uniform([self.dict_size, self.embedding_dim], -1.0, 1.0),
            name="target_embeds")
        self.context_weights = tf.Variable(
            tf.truncated_normal([self.dict_size, self.embedding_dim],
                                stddev=1.0 / math.sqrt(self.embedding_dim)),
            name="context_embeds")
        self.context_bias = tf.Variable(
            tf.zeros([self.dict_size]),
            name="context_bias")
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        self.build()

    def _build(self):
        self.outputs1 = tf.nn.embedding_lookup(self.target_embeds, self.inputs1)
        self.outputs2 = tf.nn.embedding_lookup(self.context_weights, self.inputs2)
        self.true_b = tf.nn.embedding_lookup(self.context_bias, self.inputs2)
        self.neg_outputs = tf.nn.embedding_lookup(self.context_weights, self.neg_samples)
        self.neg_b = tf.nn.embedding_lookup(self.context_bias, self.neg_samples)
        
        self.link_pred_layer = BipartiteEdgePredLayer(self.embedding_dim, self.embedding_dim,
                self.placeholders, bilinear_weights=False)

        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)
        self.outputs2 = tf.nn.l2_normalize(self.outputs2, 1)
        self.neg_outputs = tf.nn.l2_normalize(self.neg_outputs, 1)
            
    def build(self):
        self._build()
        # TF graph management
        self._loss()
        self._accuracy()
        self._minimize()

        self.p_probs = self.link_pred_layer.get_probs(self.outputs1, self.outputs2)

    def _minimize(self):
        self.opt_op = self.optimizer.minimize(self.loss)


    def _loss(self):
        aff = tf.reduce_sum(tf.multiply(self.outputs1, self.outputs2), 1) + self.true_b
        neg_aff = tf.reduce_sum(tf.multiply(self.outputs1, self.neg_outputs), 1) + self.neg_b
        # xent_loss
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(aff), logits=aff)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_aff), logits=neg_aff)
        loss = tf.reduce_sum(true_xent) + tf.reduce_sum(negative_xent)

        self.loss = loss / tf.cast(self.batch_size, tf.float32)
        self.merged_loss = tf.summary.scalar('merged_loss', self.loss)
    
    
    def _accuracy(self):
        #shape: [batch_size]
        aff = tf.reduce_sum(self.outputs1 * self.outputs2, axis=1)
        neg_aff = tf.reduce_sum(self.outputs1 * self.neg_outputs, axis=1)
        self.neg_aff = tf.expand_dims(neg_aff, axis=1)
        _aff = tf.expand_dims(aff, axis=1)
        self.aff_all = tf.concat(axis=1, values=[self.neg_aff, _aff])
        size = tf.shape(self.aff_all)[1]
        _, indices_of_ranks = tf.nn.top_k(self.aff_all, k=size)
        _, self.ranks = tf.nn.top_k(-indices_of_ranks, k=size)
        self.mrr = tf.reduce_mean(tf.div(1.0, tf.cast(self.ranks[:, -1] + 1, tf.float32)))
        self.merged_mrr = tf.summary.scalar('merged_mrr', self.mrr)
