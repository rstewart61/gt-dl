import sys
import os
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizer, RobertaModelWithHeads
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

MAX_LENGTH=80

def get_num_params(model):
    return model.num_parameters(False)
    # https://github.com/huggingface/transformers/issues/1479
    #num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #return num_params

if len(sys.argv) == 1:
    print('Arguments required: <model dir> <test dataset>')
    sys.exit(1)

#cwd = os.path.basename(os.path.normpath(os.getcwd()))
cwd = os.path.normpath(os.getcwd())

config = RobertaConfig.from_pretrained(
    "roberta-base",
    num_labels=MAX_LENGTH,
    max_length=MAX_LENGTH,
    truncation=True,
    padding="max_length"
)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base", config=config)
model = RobertaModelWithHeads.from_pretrained('roberta-base', config=config)
base_num_params = get_num_params(model)
print('Number of parameters in roberta-base:', base_num_params)

# Change 'model' to whatever directory model was saved to
#adapter_name = model.load_adapter('roberta-s2orc-cs/cs_domain_corpus')
adapter_name = model.load_adapter(sys.argv[1])
model.set_active_adapters(adapter_name)
total_num_params = get_num_params(model)
adapter_num_params = total_num_params - base_num_params
print('Number of parameters in MLM domain adapter:', adapter_num_params, adapter_num_params / base_num_params)

model.freeze_model(True)

from datasets import load_from_disk, load_metric

test_dataset = load_from_disk(sys.argv[2])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

eval_args = TrainingArguments(
    per_device_eval_batch_size=100,
    prediction_loss_only=True,
    output_dir='/tmp/eval_out',
    # The next line is important to ensure the dataset labels are properly passed to the model
    #remove_unused_columns=False,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=eval_args,
    eval_dataset=test_dataset,
    data_collator=data_collator
)

'''
metric = load_metric("accuracy")

def compute_metrics(logits, labels):
    #logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
'''

import math

eval_results = trainer.evaluate()
loss = eval_results['eval_loss']
out = open('results.txt', 'wt')
print(base_num_params, adapter_num_params, loss, math.exp(loss), file=out)
