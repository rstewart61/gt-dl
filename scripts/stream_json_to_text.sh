#!/bin/bash

INPUT_DIR=/home/brandon/S2ORC_sample/
SCRIPT=/home/brandon/5tb/src/stream_json_to_text.py
INPUT_FILE=/home/brandon/S2ORC_raw_sample.txt

if [ "$#" -lt 1 ]; then
    echo "Please specify domain: Medicine, 'Computer Science', and/or Biology"
    exit 1
fi

#(zcat $INPUT_DIR/*0 & zcat $INPUT_DIR/*2 & zcat $INPUT_DIR/*4 & zcat $INPUT_DIR/*6 & zcat $INPUT_DIR/*8 & zcat $INPUT_DIR/*a & zcat $INPUT_DIR/*c & zcat $INPUT_DIR/*f ) | python3 $SCRIPT "${@:1}"
#zcat `find $INPUT_DIR -maxdepth 1 -type f | grep -v "\."` | python3 $SCRIPT "${@:1}"
cat $INPUT_FILE | python3 $SCRIPT "${@:1}"

