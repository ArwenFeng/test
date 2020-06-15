#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Arwen'
__mtime__ = '2020/6/6'
"""
import os
import glob
import shutil
import random
import argparse
import numpy as np
from util import audio
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

def preprocess(args):
    executor = ProcessPoolExecutor(max_workers=args.num_workers)
    wav_files = glob.glob(args.wav_path)
    for wav_file in wav_files:
        executor.submit(preprocess_utterance, wav_file, args.input_path,args.output_path)
        if args.language == 'l1':
            phn_file = wav_file.replace('.wav','.PHN')
            txt_file = wav_file.replace('.wav','.TXT')
            executor.submit(copy_file,phn_file, args.input_path, args.output_path)
            executor.submit(copy_file,txt_file, args.input_path, args.output_path)
        elif args.language == 'l2':
            pass

def copy_file(origin_file, input_path, output_path):
    origin_path, name = os.path.split(origin_file)
    out_dir = origin_path.replace(input_path,output_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    target_file = os.path.join(out_dir,name)
    shutil.copy(origin_file,target_file)
    print("Copy from %s to %s"%origin_file,target_file)

def preprocess_utterance(wav_file,input_path, output_path):
    wav = audio.load_wav(wav_file)
    wav_path, name = os.path.split(wav_file)
    out_dir = wav_path.replace(input_path,output_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
    mel_filename = name.replace('.wav','.npy')
    np.save(os.path.join(out_dir, mel_filename),mel_spectrogram.T,allow_pickle=False)
    print(mel_filename,mel_spectrogram.shape[1])

# generate train_file and test_file
def train_test_split():
    wav_list = glob.glob('data/TRAIN/*/*/*.npy')
    wav_list = [wav + "\n" for wav in wav_list]
    random.shuffle(wav_list)
    print("Train files: %d" % len(wav_list))
    with open("data/train_files.txt", "w") as f:
        f.writelines(wav_list)

    wav_list = glob.glob('data/TEST/*/*/*.npy')
    wav_list = [wav + "\n" for wav in wav_list]
    random.shuffle(wav_list)
    print("TEST files: %d" % len(wav_list))
    with open("data/test_files.txt", "w") as f:
        f.writelines(wav_list)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', default='data')
    parser.add_argument('--input_path', default='/data/fyq/TIMIT')
    parser.add_argument('--wav_path', default='/data/fyq/TIMIT/*/*/*/*.wav')
    parser.add_argument('--language', default='l1', choices=['l1','l2'])
    parser.add_argument('--num_workers', type=int, default=cpu_count())
    args = parser.parse_args()
    preprocess(args)
    train_test_split()

if __name__=="__main__":
    main()
