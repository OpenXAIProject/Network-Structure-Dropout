from layers import *
import tensorflow as tf
from tensorflow.contrib.distributions import RelaxedBernoulli
import numpy as np

digamma = tf.digamma
from digamma import digamma_approx as digamma_approx
lgamma = tf.lgamma
Euler = 0.577215664901532

def bbdropout(x, training,
        alpha=1e-4, thres=1e-2, a_init=-1., tau=1e-1, center_init=1.0,
        approx_digamma=True, scale_kl=None, dep=False,
        unit_scale=True, collect=True,
        name='bbdropout', reuse=None):

    N = tf.shape(x)[0]
    K = x.shape[1].value
    is_conv = len(x.shape)==4

    with tf.variable_scope(name+'/qpi_vars', reuse=reuse):
        with tf.device('/cpu:0'):
            a = softplus(tf.get_variable('a_uc', shape=[K],
                initializer=tf.constant_initializer(a_init)))
            b = softplus(tf.get_variable('b_uc', shape=[K]))

    _digamma = digamma_approx if approx_digamma else digamma
    kl = (a-alpha)/a * (-Euler - _digamma(b) - 1/b) \
            + log(a*b) - log(alpha) - (b-1)/b
    pi = (1 - tf.random_uniform([K])**(1/b))**(1/a) if training else \
            b*tf.exp(lgamma(1+1/a) + lgamma(b) - lgamma(1+1/a+b))

    def hard_sigmoid(x):
        return tf.clip_by_value(x, thres, 1-thres)

    if dep:
        with tf.variable_scope(name+'/pzx_vars', reuse=reuse):
            hid = global_avg_pool(x) if is_conv else x
            hid = tf.stop_gradient(hid)
            with tf.device('/cpu:0'):
                hid = layer_norm(hid, scale=False, center=False)
                scale = tf.get_variable('scale', shape=[1 if unit_scale else K],
                        initializer=tf.ones_initializer())
                center = tf.get_variable('center', shape=[K],
                        initializer=tf.constant_initializer(center_init))
            hid = scale*hid + center
        if training:
            pi = pi * hard_sigmoid(hid + tf.random_normal(shape=tf.shape(hid)))
            z = RelaxedBernoulli(tau, logits=logit(pi)).sample()
        else:
            pi = pi * hard_sigmoid(hid)
            z = tf.where(tf.greater(pi, thres), pi, tf.zeros_like(pi))
        #n_active = tf.reduce_mean(
        #        tf.reduce_sum(tf.cast(tf.greater(pi, thres), tf.int32), 1))
        n_active = tf.reduce_sum(tf.cast(tf.greater(pi, thres), tf.int32), 1)
        n_active = tf.reduce_sum(n_active)/N
    else:
        if training:
            z = RelaxedBernoulli(tau, logits=logit(pi)).sample(N)
        else:
            pi_ = tf.where(tf.greater(pi, thres), pi, tf.zeros_like(pi))
            z = tf.tile(tf.expand_dims(pi_, 0), [N, 1])
        n_active = tf.reduce_sum(tf.cast(tf.greater(pi, thres), tf.int32))

    if scale_kl is None:
        kl = tf.reduce_sum(kl)
    else:
        kl = scale_kl * tf.reduce_mean(kl)

    if collect:
        if reuse is not True:
            tf.add_to_collection('kl', kl)
        prefix = 'train_' if training else 'test_'
        tf.add_to_collection(prefix+'pi', pi)
        tf.add_to_collection(prefix+'n_active', n_active)

    z = tf.reshape(z, ([-1, K, 1, 1] if is_conv else [-1, K]))
    return x*z
