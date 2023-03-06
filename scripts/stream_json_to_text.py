
import os
import json
from nltk import tokenize
import sys
import unicodedata

if len(sys.argv) == 0:
    print('Specify domains, i.e. "Computer Science" "Biology"')
    sys.exit(1)


#DOMAINS=['Medicine']
INPUT_FILE='/home/brandon/5tb/s2orc_raw.txt'
SHARD=int(sys.argv[1])
NUM_PROCS=int(sys.argv[2])
DOMAINS=sys.argv[3:]
OUTPUT_FILE='train.txt.%02d' % (SHARD)
SAVE_FILE=OUTPUT_FILE + '.save'

input_file = open(INPUT_FILE, 'rb')
if os.path.exists(SAVE_FILE):
    save_file = open(SAVE_FILE, 'rt')
    offset = save_file.read()
    save_file.close()
    print('Starting point: ', offset)
    input_file.seek(int(offset))
    output = open(OUTPUT_FILE, 'at', buffering=1024*1024)
else:
    output = open(OUTPUT_FILE, 'wt', buffering=1024*1024)
    
print('Sending output to %s' % (OUTPUT_FILE))
print('Domains are: ', DOMAINS)
print('Shard %d / %d' % (SHARD, NUM_PROCS))

def save_bytes_read():
    num_bytes = input_file.tell()
    save_file = open(SAVE_FILE + '.tmp', 'wt')
    #print('writing ', num_bytes)
    save_file.write(str(num_bytes))
    save_file.close()
    os.rename(SAVE_FILE + '.tmp', SAVE_FILE)

def is_latin(text):
    chars_remaining = 10
    for ch in text:
        if chars_remaining == 0:
            return False
        chars_remaining -= 1
        try:
            unicode_name = unicodedata.name(ch)
        except ValueError as err:
            continue
        if unicode_name.startswith('LATIN'):
            return True

input_line = 0
parsed_lines = 0
parsed_documents = 0
parse_errors = 0
for line in input_file:
  input_line += 1
  if input_line % NUM_PROCS != SHARD:
    continue
  try:
    json_dict = json.loads(line)
    fields_of_study = json_dict['fieldsOfStudy']
    skip_paper = True
    for domain in DOMAINS:
        if domain in fields_of_study:
            skip_paper = False
    if skip_paper:
        continue
    parsed_documents += 1
    abstract = json_dict['paperAbstract'].replace('\n', '').replace('\r', '').strip()
    if not is_latin(abstract):
        continue
    sample = ''
    for sentence in tokenize.sent_tokenize(abstract):
        sample += ' ' + sentence
        # Heuristic: 500 characters is about 80 words
        if len(sample) > 500:
            print(sample, file=output)
            sample = ''
            save_bytes_read()
            parsed_lines += 1
            if parsed_lines % 100000 == 0:
              print('samples', parsed_lines, 'documents', parsed_documents, 'documents', parse_errors, 'errors', input_line, 'total lines')
    if parsed_lines * NUM_PROCS > 100000000:
        print('Finished processing ~100m sentences')
        output.close()
        sys.exit(0)
        
  except json.JSONDecodeError as err:
    parse_errors += 1
  except TypeError as err2:
    parse_errors += 1
    

