#!/bin/bash

tail -10000 train.txt > test.10000.txt

for i in `seq 3 7`; do
    NUM_LINES=$((10 ** $i))
    head -$NUM_LINES train.txt > train.$NUM_LINES.txt
done

