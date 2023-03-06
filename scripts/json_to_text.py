INPUT_DIR='/home/brandon/Datasets/S2ORC/downloads/'
OUTPUT_DIR='/home/brandon/5tb/s2orc_medicine/sentences/'
DOMAINS=['Medicine']

import os
import json
import gzip
import sys
from nltk import tokenize
import argparse

def get_lines_from_file(file_path, output):
  with gzip.open(file_path, encoding="utf-8", mode='rt') as f:
    parse_errors = 0
    parsed_lines = 0
    for line in f.read().splitlines():
      try:
        json_dict = json.loads(line)
        parsed_lines += 1
        fields_of_study = json_dict['fieldsOfStudy']
        skip_paper = False
        for domain in DOMAINS:
            if domain not in fields_of_study:
                skip_paper = True
        if skip_paper:
            continue
        abstract = json_dict['paperAbstract'].replace('\n', '').replace('\r', '').strip()
        for sentence in tokenize.sent_tokenize(abstract):
            print(sentence, file=output)
            
      except json.JSONDecodeError as err:
        parse_errors += 1
      except TypeError as err2:
        parse_errors += 1
      
    print('Parsed ', parsed_lines, 'lines, found ', parse_errors, ' errors.')

# BASE_DIR + 'json_gzipped'

def merge_dataset(input_path, output_file, shard, total_shards):
    output_filename = '%s.txt.%02d' % (output_file, shard)
    output = open(output_filename, 'wt')
    print('%d: Sending output to %s' % (shard, output_filename), output_filename)
    
    for json_gzipped in os.listdir(input_path):
        if '.' in json_gzipped:
            continue
        if len(json_gzipped) != 64:
            continue
        file_as_int = int(json_gzipped, 16)
        if file_as_int % total_shards == shard:
            print(json_gzipped)
            get_lines_from_file(input_path + json_gzipped, output)
        
parser = argparse.ArgumentParser('Argument_Parser')
parser.add_argument("shard", help='current shard', type=int)
parser.add_argument("total_shards", help='total shards, i.e., #procs on computer', type=int)

args = parser.parse_args()
print(f'Shard {args.shard} out of {args.total_shards}')


merge_dataset(INPUT_DIR, OUTPUT_DIR + 'train', args.shard, args.total_shards)

