#!/usr/bin/env bash

source env_bert/bin/activate


export MODEL=bert-base-uncased
export TASK_NAME=SST2
export EXP_NAME=last3
export SEED=1
export CKPT=-1
export BS=2000


# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python ft_ds_head_wise_linear_senteval.py \
--task $TASK_NAME \
--model_name $MODEL \
--exp_name $EXP_NAME \
--seed $SEED \
--ckpt $CKPT \
--batch_size $BS

