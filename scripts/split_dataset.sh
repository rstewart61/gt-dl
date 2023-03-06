#!/bin/bash

mkdir -p train
rm -rf train/train*
split -l 200000 train.txt train/train
