#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Arwen'
__mtime__ = '2020/6/8'
"""
import glob
import shutil
import os
import numpy as np

# l2-arctic 整形
# folders = glob.glob('/data/fyq/l2-arctic/*')
# for folder in folders:
#     folder_name = os.path.split(folder)[1]
#     new_folder = folder.replace('l2-arctic', 'L2-ARCTIC')
#     os.makedirs(new_folder,)
#     for file in glob.glob(os.path.join(folder,"annotation")+'/*'):
#         name = os.path.splitext(os.path.split(file)[1])[0]
#
#         wav_file = os.path.join(os.path.join(folder,"wav"),name+'.wav')
#         txt_file = os.path.join(os.path.join(folder,"transcript"),name+'.txt')
#
#         target_wav_file = os.path.join(new_folder,folder_name+'_'+name+'.wav')
#         target_txt_file = os.path.join(new_folder,folder_name+'_'+name+'.lab')
#         shutil.copy(wav_file,target_wav_file)
#         shutil.copy(txt_file,target_txt_file)
#         print(target_wav_file,target_txt_file)

print(np.average(np.asarray([[1.1,2.1],[1.2,2.2]]),axis=0))

