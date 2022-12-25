import glob
import logging
import os
import json
import time
from datetime import datetime
from os.path import join

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from callback.optimizater.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.progressbar import ProgressBar
from tools.common import seed_everything,json_to_text
from tools.common import init_logger, logger
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import Dataset
from transformers import WEIGHTS_NAME, BertConfig,get_linear_schedule_with_warmup,AdamW, BertTokenizer
from models.MFSGC import MFSGC
from processors.ape_seq import convert_examples_to_features
from processors.ape_seq import ner_processors as processors
from processors.ape_seq import collate_fn
from processors.utils_ner import EmotionEntityLib
from tools.finetuning_argparse import get_argparse
MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'mfsgc': (BertConfig, MFSGC, BertTokenizer),
}

from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn import metrics

def compute_metrics(predictions, label_ids, ap_flag_id_ls, ap_class_map, label_map):
    label_list = [["-"+label_map[x] for j, x in enumerate(seq) if ap_flag_id_ls[i][j] != ap_class_map['O']] for i, seq in enumerate(label_ids)]
    preds_list = [["-"+label_map[x] for j, x in enumerate(seq) if ap_flag_id_ls[i][j] != ap_class_map['O']] for i, seq in enumerate(predictions)]
    class_info = classification_report(label_list, preds_list, digits=4, output_dict=True)
    return {
        "precision": precision_score(label_list, preds_list),
        "recall": recall_score(label_list, preds_list),
        "f1": f1_score(label_list, preds_list),
    }, class_info

def compute_metrics(predictions, label_ids, label_map):
    label_list = [label_map[seq] for seq in label_ids]
    preds_list = [label_map[seq] for seq in predictions]
    class_info = metrics.classification_report(label_list, preds_list, digits=4, output_dict=True)
    acc = class_info["accuracy"]
    del class_info["accuracy"]
    return {
        "accuracy": acc,
        "precision": metrics.precision_score(label_list, preds_list, average="macro"),
        "recall": metrics.recall_score(label_list, preds_list, average="macro"),
        "f1": metrics.f1_score(label_list, preds_list, average="macro"),
    }, class_info


def write_predictions(input_file, output_file, predictions):
    """ Write Product Extraction predictions to file,
        while aligning with the input format.
    """
    with open(output_file, "w") as writer, open(input_file, "r") as f:
        example_id = 0
        for line in f:
            if line.startswith("#\tpassage"):
                writer.write(line)
            elif line == "" or line == "\n":
                writer.write(line)
                if not predictions[example_id]:
                    example_id += 1
            elif len(predictions) > example_id:
                cols = line.rstrip().split()
                # 非方面词无需预测polarity
                if cols[1] == "O":
                    cols.append("-100")
                    writer.write("\t".join(cols) + "\n")
                    continue

                if len(predictions[example_id]) == 0:
                    cols.append("-100")
                else:
                    if type(predictions[example_id][0]) == list:
                        if len(predictions[example_id][0]) == 0:
                            cols.append("O")
                        else:
                            cols.extend(predictions[example_id].pop(0))
                    else:
                        label = predictions[example_id].pop(0)
                        if label == "X":
                            label = "O"
                        cols.append(label)
                writer.write("\t".join(cols) + "\n")
            else:
                logger.warning(
                    "Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0]
                )

def train(args, train_dataset, eval_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size
                * args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
                )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    pbar = ProgressBar(n_total=len(train_dataloader), desc='Training', num_epochs=int(args.num_train_epochs))
    if args.save_steps==-1 and args.logging_steps==-1:
        args.logging_steps=len(train_dataloader)
        args.save_steps = len(train_dataloader)
    best_score = -1.0
    for epoch in range(int(args.num_train_epochs)):
        pbar.reset()
        pbar.epoch_start(current_epoch=epoch)
        for step, batch in enumerate(train_dataloader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "input_multi_fusion_adjacency_matrix":batch[3],
                      "input_emotion_samples": batch[7], "input_pos_weight_q":batch[8], "ap_polarity_labels": batch[6],
                      "ap_class_labels": batch[5]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            #student model need add loss function like Lseq-fuzzy
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            pbar(step, {'loss': loss.item()})
            writer.add_scalar('train/loss', loss.item(), epoch)
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

        #evaluate for every epoch, and save best f1 model on dev
        results = evaluate(args, model, eval_dataset, epoch)
        writer.add_scalar('eval/loss', results["loss"], epoch)
        if best_score < results["f1"]:
            best_score = results["f1"]
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "best")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)
            tokenizer.save_vocabulary(output_dir)
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to %s", output_dir)

            logger.info("\n")
            if 'cuda' in str(args.device):
                torch.cuda.empty_cache()
    return global_step, tr_loss / global_step


def evaluate(args, model, eval_dataset, prefix):
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running evaluation epoch %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    all_pred_id_ls = []
    all_label_id_ls = []
    ap_flag_id_ls = []
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    if isinstance(model, nn.DataParallel):
        model = model.module
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "input_multi_fusion_adjacency_matrix":batch[3],
                      "input_emotion_samples": batch[7], "input_pos_weight_q":batch[8], "ap_polarity_labels": batch[6],
                      "ap_class_labels": batch[5]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            # tags = model.decode(logits, batch[-1])
            tags=torch.argmax(logits, dim=1).cpu().numpy()
        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        out_label_ids = inputs['ap_polarity_labels'].cpu().numpy()
        # out_label_ids = [[out_label_ids[i, j] for j in range(out_label_ids.shape[1]) if batch[-1][i, j] == 1] for i in range(out_label_ids.shape[0])]
        all_label_id_ls.extend(out_label_ids)
        all_pred_id_ls.extend(tags)
        # # 仅对aspect位置的polarity进行计算
        # ap_flag_id_ls.extend([[word for word in seq if word != -100] for seq in batch[5].numpy()])
        pbar(step)

    assert len(all_label_id_ls) == len(all_pred_id_ls)
    # assert len(all_label_id_ls[0]) == len(all_label_id_ls[0])

    eval_metric, entity_metric = compute_metrics(all_pred_id_ls, all_label_id_ls, args.id2label)

    # # Save predictions
    # write_predictions(
    #     os.path.join(args.data_dir, "valid.txt"),
    #     os.path.join(args.output_dir, "eval_predictions.txt"),
    #     [[args.id2label[x] for x in seq] for seq in all_pred_id_ls]
    # )

    logger.info("\n")
    eval_loss = eval_loss / nb_eval_steps
    results = {f'{key}': value for key, value in eval_metric.items()}
    results['loss'] = eval_loss
    logger.info("***** Eval results epoch %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    logger.info("***** Entity results epoch %s *****", prefix)
    for key in sorted(entity_metric.keys()):
        logger.info("******* %s results ********" % key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_metric[key].items()])
        logger.info(info)
    return results


def predict(args, model, tokenizer, emotionEntLib):
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(pred_output_dir)
    test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, emotionEntLib, data_type='test')
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1, collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running prediction *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", 1)
    pred_loss = 0.0
    nb_pred_steps = 0
    all_pred_id_ls = []
    all_label_id_ls = []
    pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")

    if isinstance(model, nn.DataParallel):
        model = model.module
    for step, batch in enumerate(test_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "input_multi_fusion_adjacency_matrix":batch[3],
                      "input_emotion_samples": batch[7], "input_pos_weight_q":batch[8], "ap_polarity_labels": batch[6],
                      "ap_class_labels": batch[5]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            # tags = model.decode(logits, batch[-1])
            tags = torch.argmax(logits, dim=1).cpu().numpy()
        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
        pred_loss += tmp_eval_loss.item()
        nb_pred_steps += 1
        out_label_ids = inputs['ap_polarity_labels'].cpu().numpy()
        # out_label_ids = [[out_label_ids[i, j] for j in range(out_label_ids.shape[1]) if batch[-1][i, j] == 1] for i in range(out_label_ids.shape[0])]
        all_label_id_ls.extend(out_label_ids)
        all_pred_id_ls.extend(tags)
        # # 仅对aspect位置的polarity进行计算
        # ap_flag_id_ls.extend([[word for word in seq if word != -100] for seq in batch[5].numpy()])
        pbar(step)

    assert len(all_label_id_ls) == len(all_pred_id_ls)
    # assert len(all_label_id_ls[0]) == len(all_label_id_ls[0])

    eval_metric, entity_metric = compute_metrics(all_pred_id_ls, all_label_id_ls, args.id2label)

    # Save predictions
    # write_predictions(
    #     os.path.join(args.data_dir, "mooc.test.txt.atepc"),
    #     os.path.join(args.output_dir, "test_predictions.txt"),
    #     [[args.id2label[x] for x in seq] for seq in all_pred_id_ls]
    # )

    logger.info("\n")
    pred_loss = pred_loss / nb_pred_steps
    results = {f'{key}': value for key, value in eval_metric.items()}
    results['loss'] = pred_loss
    logger.info("***** test results*****")
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    logger.info("***** Entity results*****")
    for key in sorted(entity_metric.keys()):
        logger.info("******* %s results ********" % key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_metric[key].items()])
        logger.info(info)

def load_and_cache_examples(args, task, tokenizer, emotionEntLib, data_type='train'):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    processor = processors[task]()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached-{}_{}_{}_{}'.format(
        data_type,
        args.model_type,
        str(args.train_max_seq_length if data_type == 'train' else args.eval_max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)
        features = convert_examples_to_features(examples=examples,
                                                emotionEntLib=emotionEntLib,
                                                tokenizer=tokenizer,
                                                input_type="word",
                                                ap_class_list=processor.get_ap_labels(),
                                                ap_polarity_list=processor.get_ap_polarity(),
                                                POS_tag_list=processor.get_POS_tag(),
                                                max_seq_length=args.train_max_seq_length if data_type == 'train' \
                                                    else args.eval_max_seq_length,
                                                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                sep_token=tokenizer.sep_token,
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_multi_fusion_adjacency_matrix = torch.tensor([f.input_multi_fusion_adjacency_matrix for f in features], dtype=torch.float)
    all_POS_ids = torch.tensor([f.input_POS_ids for f in features], dtype=torch.long)
    all_pos_weight_q = torch.tensor([f.pos_weight_q for f in features], dtype=torch.float)
    all_emotion_samples = torch.tensor([f.emotion_samples for f in features], dtype=torch.long)
    all_decode_mask = torch.tensor([f.decode_mask for f in features], dtype=torch.long)
    all_ap_class_ids = torch.tensor([f.ap_class_ids for f in features], dtype=torch.long)
    all_ap_polarity_ids = torch.tensor([f.ap_polarity_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_multi_fusion_adjacency_matrix,
                            all_lens, all_POS_ids, all_decode_mask, all_emotion_samples, all_pos_weight_q, all_ap_class_ids, all_ap_polarity_ids)
    return dataset


def main():
    args = get_argparse().parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + '_{}'.format(args.model_type)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

    # 创建tensorboard输出目录，把所有模型的tensorboard单独放在一起，方便拉取查看
    # 如果args.output_dir为'output/finetune/bert',则tensorboard_path为'tensorboard/finetune/bert/CURRENT_DATETIME'
    if args.output_dir.startswith('/'):
        tensorboard_path = args.output_dir[1:]
    else:
        tensorboard_path = args.output_dir
    tensorboard_path = join('tensorboard', '/'.join(tensorboard_path.split('/')[1:]))
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    tensorboard_logging_dir = join(tensorboard_path, time_)
    global writer
    writer = SummaryWriter(tensorboard_logging_dir)
    init_logger(log_file=args.output_dir + f'/{args.model_type}-{args.task_name}-{time_}.log')
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16, )
    # Set seed
    seed_everything(args.seed)
    # Prepare NER task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    POS_tag_list = processor.get_POS_tag()
    # args.POS2id = {label: i+1 for i, label in enumerate(POS_tag_list)}
    label_list = processor.get_ap_polarity()
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)
    emotionEntLib = EmotionEntityLib()

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path,num_labels=num_labels,)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,)

    config.num_ap_polarities = len(label_list)
    config.num_POS_tag = len(POS_tag_list) + 1 #需要考虑0id所以+1
    config.tail_size = emotionEntLib.tail_size
    config.relation_size = emotionEntLib.relation_size
    config.word_max_len = emotionEntLib.word_max_len
    ap_class_map = {label: i for i, label in enumerate(processor.get_ap_labels())}
    model = model_class.from_pretrained(args.model_name_or_path, config=config, ap_class_map=ap_class_map)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, emotionEntLib, data_type='train')

        #切分数据集 8:2
        train_size = int(0.8 * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_dataset, eval_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])
        # eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='dev')

        global_step, tr_loss = train(args, train_dataset, eval_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(os.path.join(args.output_dir, "best"), do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(os.path.join(args.output_dir, "best"), config=config, ap_class_map=ap_class_map)
        model.to(args.device)
        predict(args, model, tokenizer, emotionEntLib)


if __name__ == "__main__":
    main()
