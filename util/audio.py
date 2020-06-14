#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Arwen'
__mtime__ = '2020/6/6'
"""
import os
import scipy
import librosa
import random
import numpy as np
from hparams import hparams
from util.text import text_to_sequence

def load_wav(path):
  return librosa.core.load(path, sr=hparams.sample_rate)[0]

def preemphasis(x):
  return scipy.signal.lfilter([1, -hparams.preemphasis], [1], x)

def melspectrogram(y):
  D = _stft(preemphasis(y))
  S = _amp_to_db(_linear_to_mel(np.abs(D))) - hparams.ref_level_db
  return _normalize(S)

def _normalize(S):
  return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)

_mel_basis = None

def _linear_to_mel(spectrogram):
  global _mel_basis
  if _mel_basis is None:
    _mel_basis = _build_mel_basis()
  return np.dot(_mel_basis, spectrogram)

def _build_mel_basis():
  n_fft = (hparams.num_freq - 1) * 2
  return librosa.filters.mel(hparams.sample_rate, n_fft, n_mels=hparams.num_mels)

def _stft_parameters():
  n_fft = (hparams.num_freq - 1) * 2
  hop_length = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
  win_length = int(hparams.frame_length_ms / 1000 * hparams.sample_rate)
  return n_fft, hop_length, win_length

def _stft(y):
  n_fft, hop_length, win_length = _stft_parameters()
  return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def _amp_to_db(x):
  return 20 * np.log10(np.maximum(1e-5, x))


phns = ['sil', 'err','cl', 'ao', 'ch', 'ay', 'ah', 'vcl', 'm', 'th', 'uw', 'n', 'ih', 'iy', 'y', 'v', 's', 'aa', 'ey', 'el', 'jh', 'f', 'en', 'hh', 'dx', 'l', 'g', 'ow', 'b', 't', 'eh', 'p', 'w', 'er', 'epi', 'ax', 'uh', 'oy', 'd', 'k', 'dh', 'sh', 'aw', 'ae', 'ix', 'r', 'ng', 'z', 'zh']
mapping_48 = {'ax-h':'ax','ux':'uw','axr':'er','em':'m','nx':'n','eng':'ng','hv':'hh','pcl': 'cl','tcl': 'cl','kcl': 'cl','qcl': 'cl',
              'gcl': 'vcl','bcl': 'vcl','dcl': 'vcl','h#':'sil','#h':'sil','pau':'sil','q':'sil'}
mapping_39 = {'ix': 'ih','aa':'ao','ax':'ah', 'sh':'zh','el': 'l', 'en': 'n',
             'vcl':'sil','cl':'sil','epi':'sil'}

def load_vocab():
  phn2idx = {phn: idx for idx, phn in enumerate(phns)}
  idx2phn = {idx: phn for idx, phn in enumerate(phns)}
  return phn2idx, idx2phn

def get_mfccs_and_phones(wav_file, mode='train'):
  n_fft, hop_length, win_length = _stft_parameters()
  mfccs = np.load(wav_file)
  num_timesteps = mfccs.shape[0]
  (rpath, temp) = os.path.split(wav_file)
  (name, _) = os.path.splitext(temp)
  phn_file = name + ".PHN"
  phn_file = os.path.join(rpath, phn_file)

  # get sentence
  sentence_file = phn_file.replace(".PHN", ".TXT")
  with open(sentence_file) as f:
    sentence = " ".join(f.readline().strip().split(" ")[2:])
  sentence = np.asarray(text_to_sequence(sentence), dtype=np.int32)
  # get phones
  phn2idx, idx2phn = load_vocab()
  phns = np.zeros(shape=(num_timesteps,))
  bnd_list = []
  for line in open(phn_file, 'r').read().splitlines():
    start_point, end_point, phn = line.split()
    bnd = int(start_point) // hop_length
    if phn in mapping_48.keys():
      phn = mapping_48[phn]
    phns[bnd:] = phn2idx[phn]
    bnd_list.append(bnd)
  if mode == 'train':
    if hparams.aug == True:
      mfccs = spec_augment(mfccs)
    return mfccs, sentence, phns, len(phns), len(sentence)
  else:
    return wav_file, mfccs, sentence, phns, len(phns), len(sentence)

def map_to_39_2d(a):
  phn2idx, idx2phn = load_vocab()
  for i in range(0, a.shape[0]):
    for j in range(0, a.shape[1]):
      phn = idx2phn[a[i][j]]
      if phn in mapping_39.keys():
        a[i][j] = phn2idx[mapping_39[phn]]
  return a


def map_to_39_1d(a):
  phn2idx, idx2phn = load_vocab()
  for i in range(0, a.shape[0]):
    phn = idx2phn[a[i]]
    if phn in mapping_39.keys():
      a[i] = phn2idx[mapping_39[phn]]
  return a

def spec_augment(mel_spectrogram, frequency_mask_num=1, time_mask_num=1,
                 frequency_masking_para=2, time_masking_para=5):
  tau = mel_spectrogram.shape[0]
  v = mel_spectrogram.shape[1]

  warped_mel_spectrogram = mel_spectrogram

  # Step 1 : Frequency masking
  if frequency_mask_num > 0:
    for i in range(frequency_mask_num):
      f = np.random.uniform(low=0.0, high=frequency_masking_para)
      f = int(f)
      f0 = random.randint(0, v - f)
      warped_mel_spectrogram[:, f0:f0 + f] = 0

  # Step 2 : Time masking
  if time_mask_num > 0:
    for i in range(time_mask_num):
      t = np.random.uniform(low=0.0, high=time_masking_para)
      t = int(t)
      t0 = random.randint(0, tau - t)
      warped_mel_spectrogram[t0:t0 + t, :] = 0

  return warped_mel_spectrogram
