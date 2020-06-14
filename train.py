#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Arwen'
__mtime__ = '2020/6/7'
"""
import os
import time
import math
import argparse
import tensorflow as tf
from hparams import hparams as hp
from hparams import hparams_debug_string
from util import infolog
from util.tools import ValueWindow
from util import plot
from models import create_model
from datafeeder import prepare_batch, get_train_batches, get_first_train_batches

log = infolog.log

def train(log_dir, args):
    checkpoint_path = os.path.join(log_dir, 'model.ckpt')
    log('Checkpoint path: %s' % checkpoint_path)
    log(hparams_debug_string())

    # Set up model:
    global_step = tf.Variable(0, name='global_step', trainable=False)
    audio = tf.placeholder(tf.float32, [None, None, hp.num_mels], 'audio')
    sentence = tf.placeholder(tf.int32, [None, None], 'sentence')
    targets = tf.placeholder(tf.int32, [None, None], 'targets')
    audio_length = tf.placeholder(tf.int32, [None], 'audio_length')
    sentence_length = tf.placeholder(tf.int32, [None], 'sentence_length')

    with tf.variable_scope('model') as scope:
        model = create_model(args.model,hp)
        model.initialize(audio, sentence, audio_length, sentence_length, targets)
        model.add_loss()
        model.add_acc()
        model.add_optimizer(global_step)

    # Bookkeeping:
    step = 0
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    acc_window = ValueWindow(100)
    saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=2)

    # Train!
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if args.restore_step:
            # Restore from a checkpoint if the user requested it.
            restore_path = '%s-%d' % (checkpoint_path, args.restore_step)
            saver.restore(sess, restore_path)
            log('Resuming from checkpoint: %s ' % (restore_path))
        else:
            log('Starting new training run.')
        for epoch in range(1,hp.num_epochs+1):
            log('Epoch %d start...........' % epoch)
            log('Loading training data from: %s' % hp.train_data_path)
            if epoch == 1:
                batches = get_first_train_batches(hp.train_data_path)
            else:
                batches = get_train_batches(hp.train_data_path)
            log('Generated %d batches' % (len(batches)))
            for idx, batch in enumerate(batches):
                start_time = time.time()
                batch = prepare_batch(batch)
                feed_dict = {
                    model.audio: batch[0],
                    model.sentence: batch[1],
                    model.targets: batch[2],
                    model.audio_length: batch[3],
                    model.sentence_length: batch[4]
                }
                step, loss, opt, acc, preds, targets, alignment= sess.run([global_step, model.loss, model.optimize, model.acc, model.preds, model.targets, model.alignments],feed_dict=feed_dict)
                time_window.append(time.time() - start_time)
                loss_window.append(loss)
                acc_window.append(acc)
                message = 'Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f, acc=%.05f, avg_acc=%.05f]' % (
                  step, time_window.average, loss, loss_window.average, acc, acc_window.average)
                log(message)
                if loss > 1000 or math.isnan(loss):
                    log('Loss exploded to %.05f at step %d!' % (loss, step))
                    raise Exception('Loss Exploded')

            if epoch % hp.save_per_epoch == 0:
                log('Saving checkpoint to: %s-%d-%d' % (checkpoint_path, epoch, step))
                log('Saving alignments...')
                plot.plot_alignment(alignment[0], os.path.join(log_dir, 'step-%d-align.png' % step),
                                    info='%s, step=%d, loss=%.5f' % (
                                    args.model, step, loss))
                print(targets[0])
                print(preds[0])
                saver.save(sess, checkpoint_path, global_step=step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='test', help='Name of the run. Used for logging. Defaults to model name.')
    parser.add_argument('--hp', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--model', default='SED_MDD')
    parser.add_argument('--restore_step', type=int, help='Global step to restore from checkpoint.')
    parser.add_argument('--summary_interval', type=int, default=100, help='Steps between running summary ops.')
    parser.add_argument('--checkpoint_interval', type=int, default=1000, help='Steps between writing checkpoints.')
    parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    run_name = args.name
    log_dir = hp.logdir
    os.makedirs(log_dir, exist_ok=True)
    infolog.init(os.path.join(log_dir, 'train.log'), run_name)
    hp.parse(args.hp)
    train(log_dir, args)

if __name__ == '__main__':
    main()