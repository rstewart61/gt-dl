#!/bin/bash

for TASK in chemprot citation_intent rct-20k sciie; do
	TRAIN_DIR=${TASK}_train
	TEST_DIR=${TASK}_test
	for m in `seq 2 9`; do
		reduction_factor=$((2 ** $m))
		echo REDUCTION FACTOR is $reduction_factor
		for learning_rate in 1e-3 3e-3 6e-3; do
			echo LEARNING_RATE is $learning_rate
			# Task training is very noisy, run multiple times
			for iteration in `seq 5`; do
				rm -rf $TRAIN_DIR-output
				python3 ../transformer_adapter.py $TRAIN_DIR $reduction_factor $learning_rate
				python3 ../transformer_domain_loss.py $TRAIN_DIR-model/$TRAIN_DIR/ $TEST_DIR
				echo $sample_size $reduction_factor $learning_rate `cat results.txt` >> ${TASK}_all_results.txt
			done
		done
	done
done
