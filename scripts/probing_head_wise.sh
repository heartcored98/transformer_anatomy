#!/usr/bin/env bash


export MODEL=electra-large-discriminator
export TASK_NAME=17
export DEVICE=1 #$(( $TASK_NAME % 8 -3 ))
export BS=1500

echo "Start Task:" $TASK_NAME / "at Device:" $DEVICE

/opt/conda/envs/env_bert/bin/python ../probing_head_wise_linear_senteval.py \
--task $TASK_NAME \
--model_name $MODEL \
--batch_size $BS \
--device $DEVICE


echo "All job are done!"
