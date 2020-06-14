#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Arwen'
__mtime__ = '2020/6/7'
"""
import os
import argparse
import time
import tensorflow as tf
from models import create_model
from hparams import hparams as hp
from hparams import hparams_debug_string
from datafeeder import get_test_batches, prepare_batch
from util import infolog
from util import plot
from util.tools import ValueWindow, calculate_acc, obtain_list, batch_lcs
from util.audio import map_to_39_2d, load_vocab

phn2idx, idx2phn = load_vocab()
log = infolog.log

def eval(args):
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = tf.train.latest_checkpoint(hp.logdir)
    log('Loading checkpoint: %s' % checkpoint_path)
    log(hparams_debug_string())

    # Set up model:
    audio = tf.placeholder(tf.float32, [None, None, hp.num_mels], 'audio')
    sentence = tf.placeholder(tf.int32, [None, None], 'sentence')
    targets = tf.placeholder(tf.int32, [None, None], 'targets')
    audio_length = tf.placeholder(tf.int32, [None], 'audio_length')
    sentence_length = tf.placeholder(tf.int32, [None], 'sentence_length')

    # Set up model:
    with tf.variable_scope('model') as scope:
        model = create_model(args.model, hp)
        model.initialize(audio, sentence, audio_length, sentence_length, targets)
        model.add_loss()
        model.add_acc()

    # Bookkeeping:
    time_window = ValueWindow(100)
    acc_window = ValueWindow(100)
    correct_window = ValueWindow(100)

    # Eval!
    step = 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    output_list = []
    target_list = []
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)
        log('Loading evaluate data from: %s' % hp.test_data_path)
        feature_files,batches = get_test_batches(hp.test_data_path)
        for idx, batch in enumerate(batches):
            batch = prepare_batch(batch)
            feed_dict = {
                model.audio: batch[0],
                model.sentence: batch[1],
                model.targets: batch[2],
                model.audio_length: batch[3],
                model.sentence_length: batch[4]
            }
            step = step + 1
            start_time = time.time()
            time_window.append(time.time() - start_time)

            output, target, istarget, origin_acc = sess.run([model.preds, model.targets, model.istarget, model.acc],
                                                     feed_dict=feed_dict)
            # mapping to 39
            output = map_to_39_2d(output)
            target = map_to_39_2d(target)
            origin_acc_39 = calculate_acc(istarget,output,target)
            output, target, preds, labels = obtain_list(output,target,istarget)
            acc, correct = batch_lcs(output,target)
            print(origin_acc_39, acc, correct)
            acc_window.append(acc)
            correct_window.append(correct)

            output_list.extend(preds)
            target_list.extend(labels)

            message = 'Step %-7d [%.03f sec/step, avg=%.05f, correct=%.05f]' % (
                step, time_window.average, acc_window.average, correct_window.average)
            log(message)

        plot.plot_confusion_matrix(target_list, output_list, idx2phn, args.checkpoint + ".png")
        log('Confusion matrix saved!')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='', help='Path to model checkpoint')
    parser.add_argument('--name', default='test', help='Name of the run. Used for logging. Defaults to model name.')
    parser.add_argument('--hp', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--model', default='SED_MDD')
    parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
    run_name = args.name
    os.makedirs(hp.logdir, exist_ok=True)
    infolog.init(os.path.join(hp.logdir, 'eval_new.log'), run_name)
    hp.parse(args.hp)
    eval(args)


if __name__ == '__main__':
  main()