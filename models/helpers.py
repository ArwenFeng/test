import numpy as np
import tensorflow as tf
from tensorflow.contrib.seq2seq import Helper

# Adapted from tf.contrib.seq2seq.GreedyEmbeddingHelper
class MDDHelper(Helper):
  def __init__(self, sentences, audio, output_dim):
    # sentences is [N, T_s, D_s], audio is [N, T_a, D_a]
    with tf.name_scope('MDDHelper'):
      self._batch_size = tf.shape(sentences)[0]
      self._output_dim = output_dim
      self._audio = audio

      # Use full length for every target because we don't want to mask the padding frames
      num_steps = tf.shape(self._audio)[1]
      self._lengths = tf.tile([num_steps], [self._batch_size])

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def sample_ids_shape(self):
    return tf.TensorShape([])

  @property
  def sample_ids_dtype(self):
    return np.int32

  def initialize(self, name=None):
    return (tf.tile([False], [self._batch_size]), self._audio[:, 0, :])

  def sample(self, time, outputs, state, name=None):
    return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them

  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    with tf.name_scope(name or 'MDDHelper'):
      finished = (time + 1 >= self._lengths)
      next_inputs = self._audio[:, time, :]
      return (finished, next_inputs, state)