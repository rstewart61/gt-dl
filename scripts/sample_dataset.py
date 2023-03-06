#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 11:03:23 2021

@author: brandon
"""
import glob
from datasets import Dataset, load_from_disk

dataset = load_from_disk('train_processed')

print('Dataset loaded')

point_oh_oh_one_pct_dataset = dataset.filter(lambda example, indice: indice % 100000 == 0, with_indices=True, num_proc=32)
point_oh_oh_one_pct_dataset.save_to_disk('train_processed_1k')

point_oh_one_pct_dataset = dataset.filter(lambda example, indice: indice % 10000 == 0, with_indices=True, num_proc=32)
point_oh_one_pct_dataset.save_to_disk('train_processed_10k')

point_one_pct_dataset = dataset.filter(lambda example, indice: indice % 1000 == 0, with_indices=True, num_proc=32)
point_one_pct_dataset.save_to_disk('train_processed_100k')

one_pct_dataset = dataset.filter(lambda example, indice: indice % 100 == 0, with_indices=True, num_proc=32)
one_pct_dataset.save_to_disk('train_processed_1m')

ten_pct_dataset = dataset.filter(lambda example, indice: indice % 10 == 0, with_indices=True, num_proc=32)
ten_pct_dataset.save_to_disk('train_processed_10m')
