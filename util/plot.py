#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Arwen'
__mtime__ = '2020/6/8'
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_alignment(alignment, path, info=None):
    fig, ax = plt.subplots()
    im = ax.imshow(
        alignment,
        aspect='auto',
        origin='lower',
        interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.savefig(path, format='png')

def plot_confusion_matrix(y_true, y_pred, idxtophn, name):
  labels = unique_labels(y_true, y_pred)
  print(labels)
  labels = [idxtophn[id] for id in labels]
  tick_marks = np.array(range(len(labels))) + 0.5
  cm = confusion_matrix(y_true, y_pred)
  cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  print(cm_normalized)
  plt.figure(figsize=(12, 8), dpi=120)
  ind_array = np.arange(len(labels))
  x, y = np.meshgrid(ind_array, ind_array)
  # for x_val, y_val in zip(x.flatten(), y.flatten()):
  #   c = cm_normalized[y_val][x_val]
    # if c > 0.01:
    #   plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
  # offset the tick
  plt.gca().set_xticks(tick_marks, minor=True)
  plt.gca().set_yticks(tick_marks, minor=True)
  plt.gca().xaxis.set_ticks_position('none')
  plt.gca().yaxis.set_ticks_position('none')
  plt.grid(True, which='minor', linestyle='-')
  plt.gcf().subplots_adjust(bottom=0.15)
  plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title('Normalized confusion matrix')
  plt.colorbar()
  xlocations = np.array(range(len(labels)))
  plt.xticks(xlocations, labels, rotation=90)
  plt.yticks(xlocations, labels)
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  np.set_printoptions(precision=2)
  # show confusion matrix
  plt.savefig(name, format='png')
  plt.show()
