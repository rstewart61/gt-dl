import urllib.request
import os
from datasets import load_dataset, Dataset
import json
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

SAVE_DIR='/home/brandon/5tb/tasks/'

# From https://github.com/allenai/dont-stop-pretraining/blob/master/environments/datasets.py
DATASETS = {
    "chemprot": {
        "data_dir": "https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/chemprot/",
        "dataset_size": 4169
    },
    "rct-20k": {
        "data_dir": "https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/rct-20k/",
        "dataset_size": 180040
    },
    "rct-sample": {
        "data_dir": "https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/rct-sample/",
        "dataset_size": 500
    },
    "citation_intent": {
        "data_dir": "https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/citation_intent/",
        "dataset_size": 1688
    },
    "sciie": {
        "data_dir": "https://s3-us-west-2.amazonaws.com/allennlp/dont_stop_pretraining/data/sciie/",
        "dataset_size": 3219
    }
}

'''Based on https://github.com/huggingface/notebooks/blob/master/examples/language_modeling.ipynb'''
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def encode(examples):
    return tokenizer(examples['text'], max_length=MAX_LENGTH, truncation=True, padding='max_length')

for dataset, metadata in DATASETS.items():
    url = metadata['data_dir']
    labels = set()
    for env in ['dev', 'train', 'test']:
        env_url = url + env + '.jsonl'
        dataset_name = dataset + '_' + env
        path = SAVE_DIR + dataset_name + '.jsonl'
        print(dataset, path, env_url)
        if not os.path.exists(path):
            urllib.request.urlretrieve(env_url, path)
        
        text_data = {'text': []}
        sample = ''
        for line in open(path, 'rt').readlines():
            row = json.loads(line)
            labels.add(row['label'])
            text = row['text'].replace('\n', '').replace('\r', '').strip()
            sample += ' ' + text
            if len(sample) > 400:
                text_data['text'].append(sample)
                #print(sample)
                sample = ''
        json_dataset = Dataset.from_dict(text_data)
        json_dataset = json_dataset.map(encode, num_proc=32, batched=True)
        json_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        json_dataset.save_to_disk(SAVE_DIR + dataset_name)
        
    print('labels', labels)
    with open(SAVE_DIR + dataset + '_labels.txt', 'wt') as f:
        labels = [l for l in labels]
        labels.sort()
        for label in labels:
            print(label, file=f)
        f.close()
    
    
