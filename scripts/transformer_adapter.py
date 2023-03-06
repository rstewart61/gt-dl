import os
import sys
import glob

MAX_LENGTH=80

if len(sys.argv) < 3:
    print('Required arguments: <dataset> <reduction factor> <learning rate>')
    sys.exit(1)

#cwd = os.path.basename(os.path.normpath(os.getcwd()))
cwd = os.path.normpath(os.getcwd())

BASE_DIR=cwd #'/home/brandon_stewart_941/'
DATASET=sys.argv[1]
REDUCTION_FACTOR=int(sys.argv[2])
LEARNING_RATE=float(sys.argv[3])
DATASET_PATH=BASE_DIR + '/' + DATASET
OUTPUT_DIR=DATASET_PATH + '-output'
MODEL_DIR=DATASET_PATH + '-model'

from datasets import load_from_disk
train_dataset = load_from_disk(DATASET_PATH)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

'''
TEST_DATASET_PATH=''
test_dataset_pattern='*-processed-test'
for dir in glob.glob(test_dataset_pattern):
    TEST_DATASET_PATH = dir
    break
print('Test dataset is ', TEST_DATASET_PATH, 'from', test_dataset_pattern)

test_dataset = load_from_disk(TEST_DATASET_PATH)
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
'''


import transformers
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

from transformers import RobertaConfig, RobertaModelWithHeads

config = RobertaConfig.from_pretrained(
    "roberta-base",
    num_labels=MAX_LENGTH,
    max_length=MAX_LENGTH,
    truncation=True,
    padding='max_length'
)
model = RobertaModelWithHeads.from_pretrained(
    "roberta-base",
    config=config
)

adapter_name = DATASET
model.add_masked_lm_head(adapter_name)
adapter_config = transformers.PfeifferConfig(
        reduction_factor=REDUCTION_FACTOR,
        non_linearity='relu'
)
model.add_adapter(adapter_name, config=adapter_config)
model.set_active_adapters(adapter_name)
model.train_adapter(adapter_name)

import numpy as np
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction, DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

BATCH_SIZE=142 #max possible with fp16, reduction factor of 2
#BATCH_SIZE=163 #max possible with fp16, reduction factor of 7
#BATCH_SIZE=110 #max possible without fp16, reduction factor of 7

training_args = TrainingArguments(
    learning_rate=LEARNING_RATE, # 6e-4 works well with batch size 32
    num_train_epochs=1,
    per_device_train_batch_size=BATCH_SIZE,
    fp16=True,
    per_device_eval_batch_size=100,
    gradient_accumulation_steps=1,
    logging_steps=10,
    output_dir=OUTPUT_DIR,
    prediction_loss_only=False,
    overwrite_output_dir=True,
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=False,
)

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

train_metrics = trainer.train()
trainer.save_model(MODEL_DIR)

trainer.log_metrics('train', train_metrics.metrics)
print("Train ", train_metrics)
