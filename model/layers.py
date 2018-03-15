import tensorflow as tf
import numpy as np

exp = tf.exp
log = lambda x: tf.log(x + 1e-20)
logit = lambda x: log(x) - log(1-x)
softplus = tf.nn.softplus
softmax = tf.nn.softmax
tanh = tf.nn.tanh
relu = tf.nn.relu
sigmoid = tf.nn.sigmoid

dense = tf.layers.dense
flatten = tf.contrib.layers.flatten

def conv(x, filters, kernel_size=3, strides=1, **kwargs):
    return tf.layers.conv2d(x, filters, kernel_size, strides,
            data_format='channels_first', **kwargs)

def pool(x, **kwargs):
    return tf.layers.max_pooling2d(x, 2, 2,
            data_format='channels_first', **kwargs)

def global_avg_pool(x):
    return tf.reduce_mean(x, axis=[2, 3])

batch_norm = tf.layers.batch_normalization
layer_norm = tf.contrib.layers.layer_norm
