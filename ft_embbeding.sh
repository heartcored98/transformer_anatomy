#!/usr/bin/env bash

source env_bert/bin/activate


export MODEL=bert-base-uncased
export TASK_NAME=STSBenchmark
export EXP_NAME=last2
export SEED=4
export CKPT=-1
export BS=2000

# Calculate Initial Embedding
python ft_ds_head_wise_linear_senteval.py \
--task $TASK_NAME \
--model_name $MODEL \
--exp_name $EXP_NAME \
--seed $SEED \
--ckpt $CKPT \
--batch_size $BS \
--device $device \
--head 0 \
--layer 0

function head_line () {
    device=$((${1}%8))

    echo "Calculating Head ${1} Using device No. ${device}"

    python ft_ds_head_wise_linear_senteval.py \
    --task $TASK_NAME \
    --model_name $MODEL \
    --exp_name $EXP_NAME \
    --seed $SEED \
    --ckpt $CKPT \
    --batch_size $BS \
    --device $device \
    --head ${1}
}

for i in {0..11}
do
    head_line $i &

done

wait

echo "All job are done!"