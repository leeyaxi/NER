CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/outputs/conll_output/bert/best
export DATA_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="conll"
TRAIN_TYPE="teacher"
#
CUDA_VISIBLE_DEVICES=2 python run_ner_crf.py \
  --model_type=bert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --train_type=$TRAIN_TYPE \
  --do_train \
  --do_predict \
  --do_lower_case \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=128 \
  --per_gpu_train_batch_size=128 \
  --per_gpu_eval_batch_size=128 \
  --learning_rate=1e-5 \
  --num_train_epochs=100.0 \
  --logging_steps=-1 \
  --save_steps=-1 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42
