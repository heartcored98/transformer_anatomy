#!/usr/bin/env bash

source env_bert/bin/activate


export MODEL=bert-large-uncased
export TASK_NAME=MRPC # STSBenchmark SST2
export EXP_NAME=12demo
export SEED=4
export CKPT=-1
export BS=2000

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

for i in 0 #{0..11}
do
    head_line $i &
    sleep 10
done
wait
echo "All job are done!"



export MODEL=bert-large-uncased
export TASK_NAME=STSBenchmark # STSBenchmark SST2
export EXP_NAME=12demo
export SEED=3
export CKPT=-1
export BS=2000

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

for i in 0 #{0..11}
do
    head_line $i &
    sleep 10
done
wait
echo "All job are done!"


export MODEL=bert-large-uncased
export TASK_NAME=SST2 # STSBenchmark SST2
export EXP_NAME=12demo
export SEED=0
export CKPT=-1
export BS=2000

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

for i in 0 #{0..11}
do
    head_line $i &
    sleep 10
done
wait
echo "All job are done!"