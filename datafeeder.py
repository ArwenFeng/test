#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Arwen'
__mtime__ = '2020/6/7'
"""
import random
import numpy as np
from hparams import hparams as hp
from util.audio import get_mfccs_and_phones


def get_test_batches(data_path):
    with open(data_path) as f:
        lines = f.readlines()
    test_feature_files = [line.strip() for line in lines]
    n = hp.batch_size
    mod = len(test_feature_files) %n
    # Read a group of examples:
    examples = [get_next_test_example(test_feature_files,i) for i in range(len(test_feature_files))]
    feature_files = [example[0] for example in examples]
    feature_files = [feature_files[i:i+n] for i in range(0, len(examples)-mod,n)]
    examples = [example[1:] for example in examples]
    batches = [examples[i:i+n] for i in range(0, len(examples)-mod,n)]
    return feature_files,batches

def get_next_test_example(test_feature_files,i):
    '''Loads a single example (wav_file, mfccs, sentence, phns, audio_length, sentence_length) from disk'''
    feature_file = test_feature_files[i]
    return get_mfccs_and_phones(feature_file,mode='eval')


def get_first_train_batches(data_path):
    with open(data_path) as f:
        lines = f.readlines()
    feature_files = [line.strip() for line in lines]
    n = hp.batch_size
    mod = len(feature_files) % n
    examples = [get_next_train_example(feature_files,len(feature_files)) for i in range(len(feature_files))]
    examples.sort(key=lambda x: x[-1]) # for first epoch
    batches = [examples[i:i + n] for i in range(0, len(examples) - mod, n)]
    random.shuffle(batches)
    return batches

def get_train_batches(data_path):
    with open(data_path) as f:
        lines = f.readlines()
    feature_files = [line.strip() for line in lines]
    n = hp.batch_size
    mod = len(feature_files) %n
    examples = [get_next_train_example(feature_files,len(feature_files)) for i in range(len(feature_files))]
    batches = [examples[i:i+n] for i in range(0, len(examples)-mod,n)]
    random.shuffle(batches)
    return batches

def get_next_train_example(feature_files,length):
    '''Loads a single example (mfccs, sentence, phns, audio_length, sentence_length) from disk'''
    i = random.randint(0,length-1)
    feature_file = feature_files[i]
    return get_mfccs_and_phones(feature_file)

def prepare_batch(batch):
    audio = _prepare_inputs([x[0] for x in batch])
    sentence = _prepare_targets([x[1] for x in batch])
    targets = _prepare_targets([x[2] for x in batch])
    audio_length = np.asarray([x[3] for x in batch], dtype=np.int32)
    sentence_length = np.asarray([x[4] for x in batch], dtype=np.int32)
    return (audio, sentence, targets, audio_length, sentence_length)

def _prepare_targets(targets):
    max_len = max((len(x) for x in targets))
    return np.stack([_pad_1(x, max_len) for x in targets])


def _prepare_inputs(inputs):
    max_len = max((len(t) for t in inputs))
    return np.stack([_pad_2(t, max_len) for t in inputs])


def _pad_1(x, length):
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=0)


def _pad_2(t, length):
    return np.pad(t, [(0, length - t.shape[0]), (0,0)], mode='constant', constant_values=0)

