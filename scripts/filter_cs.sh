#!/bin/bash

save_dir=$PWD

#DOMAIN="Medicine|Biology" # Also: "Computer Science"
DOMAIN="Medicine"
INPUT=/home/brandon/Datasets/S2ORC/downloads
OUTPUT=/home/brandon/5tb/s2orc_medicine

echo "Filtering domain for $DOMAIN"
mkdir -p $OUTPUT

cd $INPUT
for zip_file in *
do
    echo $zip_file | egrep "lock|json" > /dev/null
    if [ $? -eq 0 ]; then
        continue
    fi
    if [ -f $OUTPUT/$zip_file.gz ]; then
        echo "Already created $OUTPUT/$zip_file.gz"
        continue
    fi
    echo $zip_file " -> " $OUTPUT/$zip_file.gz
    zcat $zip_file | egrep "$1" | gzip > $OUTPUT/$zip_file.gz
done

cd $save_dir
