CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/pretrain_model/bert-base-uncased
export DATA_DIR=$CURRENT_DIR/datasets/laptop
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="laptop"

CUDA_VISIBLE_DEVICES=2 python run_ap_class.py \
  --model_type=bert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_predict \
  --do_lower_case \
  --data_dir=$DATA_DIR/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=128 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=16 \
  --learning_rate=1e-5 \
  --crf_learning_rate=1e-3 \
  --num_train_epochs=30 \
  --logging_steps=-1 \
  --save_steps=-1 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME} \
  --overwrite_output_dir \
  --seed=42
