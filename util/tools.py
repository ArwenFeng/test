#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Arwen'
__mtime__ = '2020/6/7'
"""
import numpy as np

class ValueWindow():
  def __init__(self, window_size=100):
    self._window_size = window_size
    self._values = []

  def append(self, x):
    self._values = self._values[-(self._window_size - 1):] + [x]

  @property
  def sum(self):
    return sum(self._values)

  @property
  def count(self):
    return len(self._values)

  @property
  def average(self):
    return self.sum / max(1, self.count)

  def reset(self):
    self._values = []

def calculate_acc(istarget, preds, targets):
  num_hits = np.sum(np.multiply(np.float32(np.equal(preds, targets)), istarget))
  num_targets = np.sum(istarget)
  acc = num_hits / num_targets
  return acc

def target_seq(seq):
  first = seq[0]
  result = [first]
  for ele in seq:
    if not ele == first:
      result.append(ele)
      first = ele
  return result

def obtain_list(output, target, istarget):
  output_list = []
  target_list = []
  preds = []
  labels = []
  for i in range(output.shape[0]):
    len = int(np.sum(istarget[i]))
    o = output[i][:len]
    t = target[i][:len]
    preds.extend(o)
    labels.extend(t)
    output_list.append(target_seq(o))
    target_list.append(target_seq(t))
  return output_list, target_list, preds, labels

def batch_lcs(outputs, targets):
  result = np.asarray([list(lcs_acc(outputs[i],targets[i])) for i in range(len(outputs))])
  result = np.average(result,axis=0)
  return result

def lcs_acc(outputs, targets):
  length, LCS, lcs_o, lcs_t = lcs(outputs, targets)
  assert len(lcs_o) == len(lcs_t)
  n = float(len(targets))
  insert, subsitute, delete = 0, 0, 0
  for idx, p in enumerate(lcs_o):
    if p == '_':
      delete = delete + 1
    elif lcs_t[idx] == '_':
      insert = insert + 1
    elif not p == lcs_t[idx]:
      subsitute = subsitute + 1
  correct = (n - subsitute - delete) / n
  acc = (n - subsitute - delete - insert) / n
  return acc, correct

def lcs(str_a, str_b):
  """
  longest common subsequence of str_a and str_b
  """
  if len(str_a) == 0 or len(str_b) == 0:
    return 0
  dp = [[0 for _ in range(len(str_b) + 1)] for _ in range(len(str_a) + 1)]
  for i in range(1, len(str_a) + 1):
    for j in range(1, len(str_b) + 1):
      if str_a[i-1] == str_b[j-1]:
        dp[i][j] = dp[i-1][j-1] + 1
      else:
        dp[i][j] = max([dp[i-1][j], dp[i][j-1]])
  i, j = len(str_a), len(str_b)
  LCS = []
  LCS_A = []
  LCS_B = []
  while i > 0 or j > 0:
      if not i > 0:
          LCS_A.append('_')
          LCS_B.append(str_b[j - 1])
          j = j -1
          continue
      if not j > 0:
          LCS_A.append(str_a[i - 1])
          LCS_B.append('_')
          i = i -1
          continue
      if str_a[i-1] == str_b[j-1]:
          if dp[i][j] == dp[i-1][j-1] + 1:
              LCS.append(str_a[i - 1])
              LCS_A.append(str_a[i - 1])
              LCS_B.append(str_b[j - 1])
              i, j = i-1, j-1
              continue
      else:
          if dp[i][j] == dp[i-1][j-1]:
              LCS_A.append(str_a[i - 1])
              LCS_B.append(str_b[j - 1])
              i, j = i - 1, j - 1
              continue
          elif dp[i][j] == dp[i-1][j]:
              LCS_A.append(str_a[i - 1])
              LCS_B.append('_')
              i, j = i-1, j
              continue
          elif dp[i][j] == dp[i][j-1]:
              LCS_A.append('_')
              LCS_B.append(str_b[j - 1])
              i, j = i, j-1
              continue

  LCS_A.reverse()
  LCS_B.reverse()
  return dp[len(str_a)][len(str_b)] ,LCS,LCS_A,LCS_B