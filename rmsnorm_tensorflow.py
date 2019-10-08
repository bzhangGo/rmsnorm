# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def rms_norm(x, eps=1e-8, p=-1., bias=False, scope=None):
    """
        Root Mean Square Layer Normalization
    :param x: input tensor, with shape [batch, ..., dimension]
    :param eps: epsilon value, default 1e-8
    :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
    :param bias: whether use bias term for RMSNorm, disabled by
        default because RMSNorm doesn't enforce re-centering invariance.
    :param scope: the variable scope
    :return: a normalized tensor, with shape as `x`
    """
    with tf.variable_scope(scope or "rms_norm"):
        layer_size = x.get_shape().as_list()[-1]

        scale = tf.get_variable("scale", [layer_size], initializer=tf.ones_initializer())
        if bias:
            offset = tf.get_variable("offset", [layer_size], initializer=tf.zeros_initializer())
        else:
            offset = 0.

        if p < 0. or p > 1.:
            ms = tf.reduce_mean(x ** 2, -1, keep_dims=True)
        else:
            partial_size = int(layer_size * p)
            partial_x, _ = tf.split(x, [partial_size, layer_size - partial_size], axis=-1)

            ms = tf.reduce_mean(partial_x ** 2, -1, keep_dims=True)

        return scale * x * tf.rsqrt(ms + eps) + offset
