import sys
import os

if len(sys.argv) < 4:
    print('Required arguments: <dataset> <reduction factor> <learning rate>')
    sys.exit(1)

DATASET=sys.argv[1]
BASE_DIR=os.path.normpath(os.getcwd())
REDUCTION_FACTOR=int(sys.argv[2])
LEARNING_RATE=float(sys.argv[3])
DATASET_PATH=BASE_DIR + '/' + DATASET
OUTPUT_DIR=DATASET_PATH + '-output'
MODEL_DIR=DATASET_PATH + '-model'

base_path = '/home/brandon_stewart_941/tasks/datasets/' + DATASET
train_path = base_path + '_train.jsonl'

from datasets import load_dataset
train_dataset = load_dataset('json', data_files=train_path, split='train[0%:90%]')
eval_dataset = load_dataset('json', data_files=train_path, split='train[90%:100%]')

from transformers import RobertaTokenizer, RobertaConfig

MAX_LENGTH=80

config = RobertaConfig.from_pretrained(
    "roberta-base",
    num_labels=MAX_LENGTH,
    max_length=MAX_LENGTH,
    truncation=True,
    padding="max_length"
)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', config=config)


def tokenize_function(examples):
    return tokenizer(examples["text"], max_length=MAX_LENGTH, truncation=True, padding='max_length')


tokenized_dataset_tr = train_dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text", "label", "metadata"])
tokenized_dataset_eval = eval_dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text", "label", "metadata"])

block_size = 128

'''Copied this code from https://github.com/huggingface/notebooks/blob/master/examples/language_modeling.ipynb'''

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


lm_tr_dataset = tokenized_dataset_tr.map(
    group_texts,
    batched=True,
    batch_size=100,
    num_proc=4,
)

lm_eval_dataset = tokenized_dataset_eval.map(
    group_texts,
    batched=True,
    batch_size=100,
    num_proc=4,
)

import transformers
from transformers import RobertaModelWithHeads, DataCollatorForLanguageModeling
model = RobertaModelWithHeads.from_pretrained('roberta-base')

adapter_name = DATASET + '-mlm'
model.add_masked_lm_head(adapter_name)
adapter_config = transformers.PfeifferConfig(
        reduction_factor=REDUCTION_FACTOR,
        non_linearity='relu'
)
model.add_adapter(adapter_name, config=adapter_config)
model.set_active_adapters(adapter_name)
model.train_adapter(adapter_name)

from transformers import TrainingArguments, AdapterTrainer

BATCH_SIZE=88 #max possible with fp16, reduction factor of 32

training_args = TrainingArguments(
    learning_rate=LEARNING_RATE, # 6e-4 works well with batch size 32
    evaluation_strategy="epoch",
    num_train_epochs=100,
    per_device_train_batch_size=BATCH_SIZE,
    fp16=True,
    per_device_eval_batch_size=BATCH_SIZE,
    #weight_decay=0.01,
    #gradient_accumulation_steps=1,
    #logging_steps=10,
    output_dir=OUTPUT_DIR,
    prediction_loss_only=False,
    overwrite_output_dir=True,
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=False,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=lm_tr_dataset,
    eval_dataset=lm_eval_dataset,
    data_collator=data_collator,
)

train_metrics = trainer.train()
trainer.save_model(MODEL_DIR)

trainer.log_metrics('train', train_metrics.metrics)
print("Train ", train_metrics)

