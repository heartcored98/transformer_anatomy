#!/usr/bin/env bash

source env_bert/bin/activate


export MODEL=bert-base-uncased
export TASK_NAME=SST2
export EXP_NAME=last3
export SEED=1
export BS=4000

function head_line () {
    device=$((${1}%8))
    echo "Calculating Head ${1} Using device No. ${device}"

    python ft_ds_head_wise_linear_senteval.py \
    --task $TASK_NAME \
    --model_name $MODEL \
    --exp_name $EXP_NAME \
    --seed $SEED \
    --ckpt ${2} \
    --batch_size $BS \
    --device $device \
    --head ${1}
}

for ckpt in 12 #16  #{1..8} # 9 17
do
    for i in {0..11}
    do
        head_line $i $ckpt &
        sleep 4
    done

    wait

done

