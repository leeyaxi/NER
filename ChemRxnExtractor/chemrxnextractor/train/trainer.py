import logging
import math
import os
import re
import time
from typing import Any, Callable, Optional
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm, trange

from transformers import Trainer
from transformers import PreTrainedModel
# from transformers import is_wandb_available
from transformers import TrainingArguments
from transformers.data.data_collator import DataCollator
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.debug_utils import DebugOption
from transformers.trainer_utils import speed_metrics, PredictionOutput, denumpify_detensorize
from transformers.utils import is_sagemaker_mp_enabled

logger = logging.getLogger(__name__)


class IETrainer(Trainer):
    """
    IETrainer is inheritated from from transformers.Trainer, optimized for IE tasks.
    """
    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics=None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        use_crf: Optional[bool]=False
    ):
        super(IETrainer, self).__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            optimizers=optimizers
        )
        self.use_crf = use_crf

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        no_decay = ["bias", "LayerNorm.weight"]
        if self.use_crf:
            crf = "crf"
            crf_lr = self.args.crf_learning_rate
            logger.info(f"Learning rate for CRF: {crf_lr}")
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (not any(nd in n for nd in no_decay)) and (crf not in n)
                    ],
                    "weight_decay": self.args.weight_decay
                },
                {
                    "params": [p for p in opt_model.crf.parameters()],
                    "weight_decay": self.args.weight_decay,
                    "lr": crf_lr
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay) and (not crf not in n)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon
        )
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps
            )
        return self.lr_scheduler

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        output = self._prediction_loop(
            eval_dataloader,
            description="Evaluation",
            metric_key_prefix=metric_key_prefix,
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output

    # def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict:
    #     eval_dataloader = self.get_eval_dataloader(eval_dataset)
    #     output = self._prediction_loop(eval_dataloader, description="Evaluation")
    #
    #     self.log(output['metrics'])
    #     # self._log(output['metrics'])
    #
    #     return output

    def predict(self, test_dataset: Dataset) -> Dict:
        test_dataloader = self.get_test_dataloader(test_dataset)

        return self._prediction_loop(test_dataloader, description="Prediction")

    def _prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`
        Works both with or without labels.
        """
        model = self.model
        batch_size = dataloader.batch_size

        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)

        model.eval()

        eval_losses: List[float] = []
        preds_ids = []
        label_ids = []
        preds_ids_for_gp = []
        user_gp_pred = False
        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(
                inputs.get(k) is not None
                for k in ["labels", "lm_labels", "masked_lm_labels"]
            )

            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]

            mask = inputs["decoder_mask"].to(torch.bool)
            if len(list(logits.shape)) > 3:
                user_gp_pred = True
                preds_ = model.decode(logits, mask=mask)
                preds_ids_for_gp.extend(preds_)
                preds_for_eval = np.array(logits.cpu().numpy(), copy=True)
                preds_ids.append(preds_for_eval)
                if inputs.get("labels") is not None:
                    label_ids.append(inputs["labels"].cpu().numpy())
                    assert len(preds_for_eval) == len(inputs["labels"])

            else:
                preds = model.decode(logits, inputs["attention_mask"], decode_mask=mask)
                preds_ids.extend(preds)
                if inputs.get("labels") is not None:
                    labels = [inputs["labels"][i, mask[i]].tolist() \
                                for i in range(inputs["labels"].shape[0])]
                    label_ids.extend(labels)
                    assert len(preds) == len(labels)
                    assert len(preds[0]) == len(labels[0])

        if self.compute_metrics is not None and \
                len(preds_ids) > 0 and \
                len(label_ids) > 0:
            metrics = self.compute_metrics(preds_ids, label_ids)
        else:
            metrics = {}

        metrics = denumpify_detensorize(metrics)
        if len(eval_losses) > 0:
            metrics[f"{metric_key_prefix}_loss"] = np.mean(eval_losses)
            # metrics['eval_loss'] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        if user_gp_pred:
            return PredictionOutput(predictions=preds_ids_for_gp, label_ids=label_ids, metrics=metrics)

        return PredictionOutput(predictions=preds_ids, label_ids=label_ids, metrics=metrics)

    def _log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
        if self.epoch is not None:
            logs["epoch"] = self.epoch
        if self.global_step is None:
            # when logging evaluation metrics without training
            self.global_step = 0
        # if is_wandb_available():
        #     if self.is_world_master():
        #         wandb.log(logs, step=self.global_step)
        output = {**logs, **{"step": self.global_step}}
        if iterator is not None:
            iterator.write(output)
        else:
            logger.info(
                {k:round(v, 4) if isinstance(v, float) else v for k, v in output.items()}
            )

