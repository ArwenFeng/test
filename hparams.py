#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Arwen'
__mtime__ = '2020/6/6'
"""
import tensorflow as tf

depth = 512
# Default hyperparameters:
hparams = tf.contrib.training.HParams(
  logdir='logdir_512_aug',
  train_data_path='data/train_files.txt',
  test_data_path='data/test_files.txt',

  # Audio:
  num_mels=80,
  num_freq=1025,
  sample_rate=16000,
  frame_length_ms=50,
  frame_shift_ms=12.5,
  preemphasis=0.97,
  min_level_db=-100,
  ref_level_db=20,
  aug = True,

  # Model:
  dropout_rate=0.5,
  embed_depth=depth,
  prenet_depths=[depth, depth//2],
  encoder_depth=depth,
  postnet_depth=depth,
  attention_depth=depth,
  decoder_depth=depth,

  # Training:
  batch_size=32,
  lr=0.0003,
  num_epochs=100,
  steps_per_epoch=100,
  save_per_epoch=10,
)


def hparams_debug_string():
  values = hparams.values()
  hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
  return 'Hyperparameters:\n' + '\n'.join(hp)