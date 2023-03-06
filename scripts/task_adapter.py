#!/usr/bin/env python3

MAX_LENGTH=80
#BASE_DIR='drive/My Drive/dl-fp/s2orc_cs/'
#BASE_DIR='/home/brandon/5tb/s2orc_cs/'
BASE_DIR='/home/brandon_stewart_941/'
OUTPUT_DIR=BASE_DIR + 'roberta-retrained'
DATASET=BASE_DIR + 'citation-intent/train.jsonl'
DOMAIN_ADAPTER_PATH='roberta-s2orc-cs/cs_domain_corpus'

import transformers
from transformers import RobertaTokenizer, pipeline
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaModelWithHeads
from datasets import load_dataset, Value, ClassLabel, Features


id_to_label = {
        0: "Background",
        1: "CompareOrContrast",
        2: "Extends",
        3: "Future",
        4: "Motivation",
        5: "Uses"
        }

classes = ["Background", "CompareOrContrast", "Extends", "Future", "Motivation", "Uses"]
num_labels = len(classes)
features = Features({
        "text":  Value('string'),
        "label": ClassLabel(num_classes=num_labels, names=classes)
        })

config = RobertaConfig.from_pretrained(
    "roberta-base",
    num_labels=num_labels,
)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base", config=config)

dataset = load_dataset('json', data_files=DATASET, features=features, split='train')

print(DATASET + ' num rows = ', dataset.num_rows)

def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")

# Encode the input data
dataset = dataset.map(encode_batch, batched=True, num_proc=4)

# The transformers model expects the target class column to be named "labels"
dataset.rename_column_("label", "labels")

# Transform to pytorch tensors and only output the required columns
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

dataset_dict = dataset.train_test_split(test_size=0.2)
print('train', dataset_dict['train'].num_rows, 'test', dataset_dict['test'].num_rows)

model = RobertaModelWithHeads.from_pretrained('roberta-base', config=config)

adapter_name = model.load_adapter(DOMAIN_ADAPTER_PATH)
model.set_active_adapters(adapter_name)
fill_mask = pipeline(
    "fill-mask",
    model=model,
    tokenizer=tokenizer
)

def test_sentence(sentence):
    print(sentence)
    result = fill_mask(sentence)
    for row in result:
        print(row['token_str'])

test_sentence("The guide also illustrates the <mask> between concurrency design patterns and the quality attributes they address")
test_sentence("software practitioners are facing the <mask> of developing high quality multi-threaded programs")


# Add a new adapter
adapter_name = 'citation-intent-task'
model.add_classification_head(adapter_name, num_labels=num_labels, id2label=id_to_label)
#model.add_masked_lm_head(adapter_name)
adapter_config = transformers.PfeifferConfig( reduction_factor=7, non_linearity='relu')
model.add_adapter(adapter_name, config=adapter_config)
model.set_active_adapters(adapter_name)
model.train_adapter(adapter_name)

import numpy as np
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    learning_rate=5e-5, # 3e-3
    #weight_decay=0.01,
    num_train_epochs=50,
    per_device_train_batch_size=163, #32
    per_device_eval_batch_size=163, #32
    logging_steps=25,
    evaluation_strategy='steps',
    output_dir=OUTPUT_DIR,
    prediction_loss_only=True,
    overwrite_output_dir=True,
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=False,
    fp16=True,
)


def compute_accuracy(p: EvalPrediction):
  preds = np.argmax(p.predictions, axis=1)
  return {"acc": (preds == p.label_ids).mean()}


trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict['train'],
    eval_dataset=dataset_dict['test'],
    compute_metrics=compute_accuracy,
)

results = trainer.train()
trainer.log_metrics('train', results.metrics)
trainer.save_model(BASE_DIR + 'roberta-citation-intent-classifier')
