#!/usr/bin/env bash

source env_bert/bin/activate


export MODEL=bert-base-uncased
export TASK_NAME=SST2
export EXP_NAME=last3
export SEED=1
export CKPT=-1
export BS=4000


python ft_ds_head_wise_linear_senteval.py \
--task $TASK_NAME \
--model_name $MODEL \
--exp_name $EXP_NAME \
--seed $SEED \
--batch_size $BS \
--device 0 \
--ckpt_run


echo "All job are done!"
