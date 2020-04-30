#!/usr/bin/env bash


export MODEL=bert-base-uncased   #electra-large-discriminator
export TASK_NAME=21 # 14 SST2  17 MRPC 21 STSB
export DEVICE=0 #$(( $TASK_NAME % 8 -3 ))
export BS=1500

echo "Start Task:" $TASK_NAME / "at Device:" $DEVICE

/opt/conda/envs/env_bert/bin/python ../ds_head_wise_linear_senteval.py \
--task $TASK_NAME \
--model_name $MODEL \
--batch_size $BS \
--device $DEVICE


echo "All job are done!"
