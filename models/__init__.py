#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Arwen'
__mtime__ = '2020/6/6'
"""
from .sed_mdd import SED_MDD

def create_model(name, hparams):
    if name == "SED_MDD":
        return SED_MDD(hparams)
    else:
        raise Exception('Unknown model: ' + name)