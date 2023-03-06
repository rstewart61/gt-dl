#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 10:09:15 2021

@author: brandon
"""

from transformers import pipeline, RobertaConfig, RobertaForMaskedLM, RobertaTokenizer, RobertaModelWithHeads

MAX_LENGTH=128


config = RobertaConfig.from_pretrained(
    "roberta-base",
    num_labels=MAX_LENGTH,
    max_length=MAX_LENGTH,
    truncation=True,
    padding="max_length"
)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base", config=config)
model = RobertaModelWithHeads(config=config)

# Change 'model' to whatever directory model was saved to
adapter_name = model.load_adapter('model')
model.set_active_adapters(adapter_name)
fill_mask = pipeline(
    "fill-mask",
    model=model,
    tokenizer=tokenizer
)

print(fill_mask("The guide also illustrates the <mask> between concurrency design patterns and the quality attributes they address"))
print(fill_mask("software practitioners are facing the <mask> of developing high quality multi-threaded programs"))
