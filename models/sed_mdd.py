#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
from models.modules import m_rnn, prenet
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper
from tensorflow.contrib.seq2seq import BasicDecoder, BahdanauAttention, AttentionWrapper
from models.rnn_wrappers import DecoderPrenetWrapper, ConcatOutputAndAttentionWrapper
from util.audio import phns
from util.infolog import log
from util.text import symbols

class SED_MDD():
    def __init__(self,hparams):
        self.hp = hparams

    def initialize(self, audio, sentence, audio_length, sentence_length, targets=None):
        with tf.variable_scope('inference') as scope:
            batch_size = self.hp.batch_size
            is_training = targets is not None
            # Embeddings
            embedding_table = tf.get_variable(
                'embedding', [len(symbols), self.hp.embed_depth], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.5))
            embedded_inputs = tf.nn.embedding_lookup(embedding_table, sentence)  # [N, T_in, embed_depth=256]

            # Encoder
            prenet_outputs = prenet(embedded_inputs, is_training, self.hp.prenet_depths)  # [N, T_in, prenet_depths[-1]=128]
            encoder_outputs = m_rnn(prenet_outputs,sentence_length,is_training,
                                    'encoder_m_rnn',self.hp.encoder_depth)# [N, T_in, encoder_depth=256]


            # Attention
            attention_cell = AttentionWrapper(
                GRUCell(self.hp.attention_depth),
                BahdanauAttention(self.hp.attention_depth, encoder_outputs),
                alignment_history=True,
                output_attention=False)

            # Apply prenet before concatenation in AttentionWrapper.
            attention_cell = DecoderPrenetWrapper(attention_cell, is_training, self.hp.prenet_depths)

            # Concatenate attention context vector and RNN cell output into a 2*attention_depth=512D vector.
            concat_cell = ConcatOutputAndAttentionWrapper(attention_cell)

            # Decoder (layers specified bottom to top):
            decoder_cell = MultiRNNCell([
                OutputProjectionWrapper(concat_cell, self.hp.decoder_depth),
                ResidualWrapper(GRUCell(self.hp.decoder_depth)),
                ResidualWrapper(GRUCell(self.hp.decoder_depth))
            ], state_is_tuple=True)  # [N, T_in, decoder_depth=256]

            # Project onto logit (predict 1 frame logits at each RNN step):
            output_cell = OutputProjectionWrapper(decoder_cell, len(phns))
            logits, final_decoder_state = tf.nn.dynamic_rnn(output_cell, audio, sequence_length=audio_length,dtype=tf.float32)

            # Grab alignments from the final decoder state:
            self.alignments = tf.transpose(final_decoder_state[0].alignment_history.stack(), [1, 2, 0])

            self.ppgs = tf.nn.softmax(logits, name='ppgs')  # (N, T, V)
            self.preds = tf.to_int32(tf.argmax(logits, axis=-1))  # (N, T)

            self.audio = audio
            self.sentence = sentence
            self.audio_length = audio_length
            self.sentence_length = sentence_length
            self.prenet_outputs = prenet_outputs
            self.cbhg_outputs = encoder_outputs
            self.logits = logits
            self.targets = targets

            log('Initialized the model. Dimensions: ')
            log('  audio:               %d' % audio.shape[-1])
            log('  prenet out:          %d' % prenet_outputs.shape[-1])
            log('  m_rnn out:            %d' % encoder_outputs.shape[-1])
            log('  logits:              %d' % logits.shape[-1])

    def add_loss(self):
        '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
        with tf.variable_scope('loss') as scope:
            istarget = tf.sign(tf.abs(tf.reduce_sum(self.audio, -1)))  # indicator: (N, T)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                  labels=self.targets)
            loss *= istarget
            self.loss = tf.reduce_mean(loss)

    def add_acc(self):
        with tf.variable_scope('acc') as scope:
            istarget = tf.sign(tf.abs(tf.reduce_sum(self.audio, -1)))  # indicator: (N, T)
            num_hits = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.targets)) * istarget)
            num_targets = tf.reduce_sum(istarget)
            self.acc = num_hits / num_targets
            self.istarget = istarget

    def add_optimizer(self, global_step):
        '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.

        Args:
          global_step: int32 scalar Tensor representing current global step in training
        '''
        with tf.variable_scope('optimizer') as scope:
            hp = self.hp
            lr = tf.get_variable('learning_rate', initializer=hp.lr, trainable=False)
            self.learning_rate = lr
            optimizer = tf.train.AdamOptimizer(lr)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            self.gradients = gradients
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
            # https://github.com/tensorflow/tensorflow/issues/1122
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
                                                          global_step=global_step)