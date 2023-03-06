#!/bin/bash

SCRIPT=/home/brandon/5tb/src/stream_json_to_text.py
INPUT_FILE=/home/brandon/5tb/s2orc_raw_sample.txt
NUM_PROCS=32

if [ "$#" -lt 1 ]; then
    echo "Please specify domain: Medicine, 'Computer Science', and/or Biology"
    exit 1
fi

for i in `seq 0 $(($NUM_PROCS-1))`; do
    python3 $SCRIPT $i $NUM_PROCS "${@:1}" &
done

wait
echo "All processes completed!!!!!!!!"
