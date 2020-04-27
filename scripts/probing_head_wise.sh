#!/usr/bin/env bash


export MODEL=electra-small-discriminator
export TASK_NAME=1
export DEVICE=$(( $TASK_NAME % 8))
export BS=4000

echo "Start Task:" $TASK_NAME / "at Device:" $DEVICE

/opt/conda/envs/env_bert/bin/python ../head_wise_linear_senteval.py \
--task $TASK_NAME \
--model_name $MODEL \
--batch_size $BS \
--device $DEVICE


echo "All job are done!"
