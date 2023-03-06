#!/bin/bash

DOMAIN=cs
#DOMAIN=biomed

echo DOMAIN is $DOMAIN

for n in `seq 4 5`; do
	sample_size=$((10 ** $n))
	echo SAMPLE SIZE is $sample_size
	for m in `seq 2 7`; do
		reduction_factor=$((2 ** $m))
		echo REDUCTION FACTOR is $reduction_factor
		for learning_rate in 1e-3 3e-3 6e-3; do
			echo LEARNING_RATE is $learning_rate
			python3 ../transformer_adapter.py s2orc_$DOMAIN-$sample_size-processed $reduction_factor $learning_rate
			python3 ../transformer_domain_loss.py s2orc_$DOMAIN-$sample_size-processed-model/s2orc_$DOMAIN-$sample_size-processed/ s2orc_$DOMAIN-10000-processed-test
			echo $sample_size $reduction_factor $learning_rate `cat results.txt` >> ${DOMAIN}_all_results.txt
		done
	done
done
