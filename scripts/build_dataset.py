import numpy as np
import torch
import os
import sys
import gzip
import json
import glob
from datasets import Dataset, load_dataset

from transformers import RobertaConfig, RobertaTokenizer

MAX_LENGTH=80

config = RobertaConfig.from_pretrained(
    "roberta-base",
    num_labels=MAX_LENGTH,
    max_length=MAX_LENGTH,
    truncation=True,
    padding="max_length"
)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base", config=config)

from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, AdapterTrainer, LineByLineTextDataset

cwd = os.path.basename(os.path.normpath(os.getcwd()))
print('cwd = ' + cwd)

def build_dataset(cur_file):
    parts = cur_file.split('.')
    amount = 'all'
    if len(parts) > 2:
        amount = parts[1]
    dataset_name = cwd + '-' + amount + '-processed'
    if parts[0] == 'test':
        dataset_name += '-test'
    if os.path.exists(dataset_name):
        print('Already processed', cur_file, 'skipping')
        return
    print('building', cur_file)
    cache_dir = '/home/brandon/5tb/cache/' + dataset_name
    dataset = load_dataset('text', data_files=cur_file, split='train', cache_dir=cache_dir)
    def encode(examples):
      return tokenizer(examples['text'], max_length=MAX_LENGTH, truncation=True, padding='max_length')
    dataset = dataset.map(encode, num_proc=32, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataset.save_to_disk(dataset_name)
    
build_dataset('test.10000.txt')
files = glob.glob('train.10*.txt')
files.sort()
for cur_file in files:
    build_dataset(cur_file)
#build_dataset('train.txt')
