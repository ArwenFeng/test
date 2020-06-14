#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Arwen'
__mtime__ = '2020/6/6'
"""
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from hparams import hparams as hp

def prenet(inputs, is_training, layer_sizes, scope=None):
    x = inputs
    drop_rate = hp.dropout_rate if is_training else 0.0
    with tf.variable_scope(scope or 'prenet'):
        for i, size in enumerate(layer_sizes):
          dense = tf.layers.dense(x, units=size, activation=tf.nn.relu, name='dense_%d' % (i+1))
          x = tf.layers.dropout(dense, rate=drop_rate, training=is_training, name='dropout_%d' % (i+1))
        return x

def conv1d(inputs, kernel_size, channels, activation, is_training, scope):
    with tf.variable_scope(scope):
        conv1d_output = tf.layers.conv1d(
          inputs,
          filters=channels,
          kernel_size=kernel_size,
          activation=activation,
          padding='same')
        return tf.layers.batch_normalization(conv1d_output, training=is_training)

def m_rnn(inputs, input_lengths,is_training, scope, depth):
    with tf.variable_scope(scope):
        conv1_output = conv1d(inputs, 5, depth, tf.nn.relu, is_training, 'proj_1')
        maxpool1_output = tf.layers.max_pooling1d(conv1_output,pool_size=2,strides=1,padding='same')
        half_depth = depth // 2
        assert half_depth * 2 == depth, 'encoder and postnet depths must be even.'

        outputs, states = tf.nn.bidirectional_dynamic_rnn(
          GRUCell(half_depth),
          GRUCell(half_depth),
          maxpool1_output,
          sequence_length=input_lengths,
          dtype=tf.float32)
        return tf.concat(outputs, axis=2)