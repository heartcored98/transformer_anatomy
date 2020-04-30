# Install newest ptl.
# pip install -U git+http://github.com/PyTorchLightning/pytorch-lightning/
# Install example requirements
# pip install -r ../requirements.txt

# Download glue data
# python3 ../../utils/download_glue_data.py

source ~/.bashrc
conda activate env_bert

export SEED=3
export CUDA_VISIBLE_DEVICES=$(( $SEED % 4 ))

export TASK=sst-2 #'cola': 2, 'mnli': 3, 'mrpc': 2, 'sst-2': 2, 'sts-b': 1, 'qqp': 2, 'qnli': 2, 'rte': 2, 'wnli': 2
export DATA_DIR=../../data/glue_data/SST-2 #/glue_data/MRPC/
export EXP_NAME=last_layer_snapshot

export MAX_LENGTH=128
export LEARNING_RATE=2e-5
export BERT_MODEL=googlebert-base-uncased
export BATCH_SIZE=32
export NUM_EPOCHS=3
export OUTPUT_DIR_NAME=${BERT_MODEL}-${TASK}-${SEED}
export OUTPUT_DIR=${PWD}/${EXP_NAME}/${OUTPUT_DIR_NAME}

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR
# Add parent directory to python path to access transformer_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

python run_pl_glue.py --data_dir $DATA_DIR \
--task $TASK \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--learning_rate $LEARNING_RATE \
--num_train_epochs $NUM_EPOCHS \
--train_batch_size $BATCH_SIZE \
--seed $SEED \
--do_train \
--do_predict \
--n_gpu 1 \
--n_worker 4 \
