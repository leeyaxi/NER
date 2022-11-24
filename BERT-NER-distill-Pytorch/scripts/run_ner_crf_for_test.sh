CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/bert-base-uncased
export DATA_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="conll"
TRAIN_TYPE="teacher"
#
CUDA_VISIBLE_DEVICES=1 python run_ner_crf_test.py \
  --model_type=bilstm \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --train_type=$TRAIN_TYPE \
  --do_predict \
  --do_lower_case \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=128 \
  --per_gpu_eval_batch_size=128 \
  --learning_rate=1e-5 \
  --num_train_epochs=100 \
  --logging_steps=-1 \
  --save_steps=-1 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42
