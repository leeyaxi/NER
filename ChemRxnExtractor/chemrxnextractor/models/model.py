from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging
import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers import BertForTokenClassification
# from .crf import ConditionalRandomField as CRF
# from .crf import allowed_transitions
from .crf_bak import CRF
from .GlobalPointer import GlobalPointer
from .pooler import Pooler
from .loss import loss_fun


logger = logging.getLogger(__name__)


class BertForTagging(BertForTokenClassification):
    def __init__(self, config, use_cls=False):
        super(BertForTagging, self).__init__(config)

        self.use_cls = use_cls
        if self.use_cls:
            self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        decoder_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        if self.use_cls:
            batch_size, seq_length, hidden_dim = sequence_output.shape
            extended_cls_h = outputs[1].unsqueeze(1).expand(batch_size, seq_length, hidden_dim)
            sequence_output = torch.cat([sequence_output, extended_cls_h], 2)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

    def decode(self, logits, mask):
        preds = torch.argmax(logits, dim=2).cpu().numpy()
        batch_size, seq_len = preds.shape
        preds_list = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            for j in range(seq_len):
                if mask[i, j]:
                    preds_list[i].append(preds[i,j])
        return preds_list


class BertCRFForTagging(BertForTokenClassification):
    def __init__(self, config, tagging_schema="BIO", use_cls=False):
        super(BertCRFForTagging, self).__init__(config)

        logger.info(f"Tagging schema: {tagging_schema}")
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)
        self.use_cls = use_cls
        if self.use_cls:
            self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        decoder_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output = outputs[0]

        if self.use_cls:
            batch_size, seq_length, hidden_dim = sequence_output.shape
            extended_cls_h = outputs[1].unsqueeze(1).expand(batch_size, seq_length, hidden_dim)
            sequence_output = torch.cat([sequence_output, extended_cls_h], 2)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss = self.crf(emissions = logits, tags=labels, mask=attention_mask)
            outputs =(-1*loss,)+outputs
        # else:
        #     prediction = self.crf.decode(logits, mask=decoder_mask)

        return outputs

    def decode(self, logits, attention_mask, decode_mask):
        # viterbi_path = self.crf.viterbi_tags(logits, mask)
        # viterbi_tags = [x for x, y in viterbi_path]
        viterbi_path = self.crf.decode(logits, attention_mask)
        viterbi_tags = viterbi_path.squeeze(0).cpu().numpy()
        decode_tags = []
        for i in range(viterbi_tags.shape[0]):
            decode_tag = []
            for j in range(viterbi_tags.shape[1]):
                if decode_mask[i, j]:
                    decode_tag.append(viterbi_tags[i, j])
            decode_tags.append(decode_tag)

        return decode_tags

class BertGlobalPointerForTagging(BertForTokenClassification):
    def __init__(self, config, tagging_schema="BIO", use_cls=False, RoPE=True):
        super(BertGlobalPointerForTagging, self).__init__(config)
        config.inner_dim = 64
        self.inner_dim = config.inner_dim
        logger.info(f"Tagging schema: {tagging_schema}")
        
        self.use_cls = use_cls
        if self.use_cls:
            self.global_pointer = GlobalPointer(hidden_size=config.hidden_size * 2, heads=config.num_labels, head_size=config.inner_dim)
        else:
            self.global_pointer = GlobalPointer(hidden_size=config.hidden_size, heads=config.num_labels, head_size=config.inner_dim)

        self.RoPE = RoPE

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        decoder_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output = outputs[0]

        if self.use_cls:
            batch_size, seq_length, hidden_dim = sequence_output.shape
            extended_cls_h = outputs[1].unsqueeze(1).expand(batch_size, seq_length, hidden_dim)
            sequence_output = torch.cat([sequence_output, extended_cls_h], 2)

        sequence_output = self.dropout(sequence_output)
        logits = self.global_pointer(sequence_output, attention_mask.gt(0).long())

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss = loss_fun(logits, labels)
            outputs = (loss,) + outputs

        return outputs

    def decode(self, logits, mask):
        mask = mask.cpu()
        pred = np.array(logits.cpu().numpy(), copy=True)
        pred[:, [0, -1]] -= np.inf
        pred[:, :, [0, -1]] -= np.inf

        batch_size, seq_len = pred.shape[0], pred.shape[1]
        preds_list = [None for _ in range(batch_size)]
        for i in range(batch_size):
            remove_ixs = np.where(mask[i] == False)
            logits_ = np.delete(pred[i, :], remove_ixs, axis=1)
            preds_list[i] = (np.delete(logits_, remove_ixs, axis=2))
        return preds_list

class BertForRoleLabeling(BertForTokenClassification):
    def __init__(self, config, use_cls=False, prod_pooler="head"):
        super(BertForRoleLabeling, self).__init__(config)
        self.use_cls = use_cls
        self.pooler = Pooler(config)
        self.prod_pool_type = prod_pooler

        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.activation = nn.Tanh()

        hidden_size = config.hidden_size*3 if self.use_cls else config.hidden_size*2
        self.classifier = nn.Linear(hidden_size, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        prod_start_mask=None,
        prod_end_mask=None,
        prod_mask=None,
        decoder_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output = outputs[0]

        if self.prod_pool_type == "span":
            prod_h = self.pooler.pool_span(sequence_output, prod_mask)
        else:
            prod_h = self.pooler.pool_head(sequence_output, prod_start_mask)

        prod_h = prod_h.unsqueeze(1)
        batch_size, seq_length, hidden_dim = sequence_output.shape
        extended_prod_h = prod_h.expand(batch_size, seq_length, hidden_dim)

        # concatenate sequence_output with Product token output
        # sequence_output = self.activation(self.dense(sequence_output))
        extended_sequence_output = torch.cat([sequence_output, extended_prod_h], 2)
        if self.use_cls:
            extended_cls_h = outputs[1].unsqueeze(1).expand(batch_size, seq_length, hidden_dim)
            extended_sequence_output = torch.cat([extended_sequence_output, extended_cls_h], 2)
        # extended_sequence_output = sequence_output
        extended_sequence_output = self.dropout(extended_sequence_output)
        logits = self.classifier(extended_sequence_output)

        loss = None
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None: # for possible speed up
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs

    def decode(self, logits, mask):
        preds = torch.argmax(logits, dim=2).cpu().numpy()
        batch_size, seq_len = preds.shape
        preds_list = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            for j in range(seq_len):
                if mask[i, j]:
                    preds_list[i].append(preds[i,j])
        return preds_list

class BertCRFForRoleLabeling(BertForTokenClassification):
    def __init__(
        self,
        config,
        tagging_schema="BIO",
        use_cls=False,
        prod_pooler="head"
    ):
        super(BertCRFForRoleLabeling, self).__init__(config)
        self.use_cls = use_cls
        self.pooler = Pooler(config)
        self.prod_pool_type = prod_pooler

        # if use_cls is True, concatenate token_repr with prod_repr and cls_repr
        hidden_size = config.hidden_size*3 if self.use_cls else config.hidden_size*2

        self.classifier = nn.Linear(hidden_size, config.num_labels)

        self.init_weights()

        logging.info(f"Tagging schema: {tagging_schema}")
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        prod_start_mask=None,
        prod_end_mask=None,
        prod_mask=None,
        decoder_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        # outputs[0]: hidden states of all time steps
        # outputs[1]: hidden state of [CLS]
        sequence_output = outputs[0]

        if self.prod_pool_type == "span":
            prod_h = self.pooler.pool_span(sequence_output, prod_mask)
        else:
            prod_h = self.pooler.pool_head(sequence_output, prod_start_mask)

        prod_h = prod_h.unsqueeze(1)
        batch_size, seq_length, hidden_dim = sequence_output.shape
        extended_prod_h = prod_h.expand(batch_size, seq_length, hidden_dim)

        # concatenate sequence_output with Product token output
        sequence_output = torch.cat([sequence_output, extended_prod_h], 2)
        if self.use_cls:
            extended_cls_h = outputs[1].unsqueeze(1).expand(batch_size, seq_length, hidden_dim)
            sequence_output = torch.cat([sequence_output, extended_cls_h], 2)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        assert logits.shape[2] == self.num_labels
        loss = None
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss = self.crf(emissions = logits, tags=labels, mask=attention_mask)
            outputs =(-1*loss,)+outputs
        return outputs

    def decode(self, logits, attention_mask, decode_mask):
        viterbi_path = self.crf.decode(logits, attention_mask)
        viterbi_tags = viterbi_path.squeeze(0).cpu().numpy()
        decode_tags = []
        for i in range(viterbi_tags.shape[0]):
            decode_tag = []
            for j in range(viterbi_tags.shape[1]):
                if decode_mask[i, j]:
                    decode_tag.append(viterbi_tags[i, j])
            decode_tags.append(decode_tag)

        return decode_tags

class BertGlobalPointerForRoleLabeling(BertForTokenClassification):
    def __init__(
        self,
        config,
        tagging_schema="BIO",
        use_cls=False,
        RoPE=True,
        prod_pooler="head"
    ):
        super(BertGlobalPointerForRoleLabeling, self).__init__(config)
        self.use_cls = use_cls
        self.pooler = Pooler(config)
        self.prod_pool_type = prod_pooler
        config.inner_dim = 64
        self.inner_dim = config.inner_dim
        # if use_cls is True, concatenate token_repr with prod_repr and cls_repr
        if self.use_cls:
            self.global_pointer = GlobalPointer(hidden_size=config.hidden_size * 3, heads=config.num_labels, head_size=config.inner_dim)
        else:
            self.global_pointer = GlobalPointer(hidden_size=config.hidden_size * 2, heads=config.num_labels, head_size=config.inner_dim)

        self.RoPE = RoPE

        self.init_weights()

        logging.info(f"Tagging schema: {tagging_schema}")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        prod_start_mask=None,
        prod_end_mask=None,
        prod_mask=None,
        decoder_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        # outputs[0]: hidden states of all time steps
        # outputs[1]: hidden state of [CLS]
        sequence_output = outputs[0]

        if self.prod_pool_type == "span":
            prod_h = self.pooler.pool_span(sequence_output, prod_mask)
        else:
            prod_h = self.pooler.pool_head(sequence_output, prod_start_mask)

        prod_h = prod_h.unsqueeze(1)
        batch_size, seq_length, hidden_dim = sequence_output.shape
        extended_prod_h = prod_h.expand(batch_size, seq_length, hidden_dim)

        # concatenate sequence_output with Product token output
        sequence_output = torch.cat([sequence_output, extended_prod_h], 2)
        if self.use_cls:
            extended_cls_h = outputs[1].unsqueeze(1).expand(batch_size, seq_length, hidden_dim)
            sequence_output = torch.cat([sequence_output, extended_cls_h], 2)
        sequence_output = self.dropout(sequence_output)

        logits = self.global_pointer(sequence_output, attention_mask.gt(0).long())

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss = loss_fun(logits, labels)
            outputs = (loss,) + outputs

        return outputs

    def decode(self, logits, mask):
        mask = mask.cpu()
        pred = np.array(logits.cpu().numpy(), copy=True)
        pred[:, [0, -1]] -= np.inf
        pred[:, :, [0, -1]] -= np.inf

        batch_size, seq_len = pred.shape[0], pred.shape[1]
        preds_list = [None for _ in range(batch_size)]
        for i in range(batch_size):
            remove_ixs = np.where(mask[i] == False)
            logits_ = np.delete(pred[i, :], remove_ixs, axis=1)
            preds_list[i] = (np.delete(logits_, remove_ixs, axis=2))
        return preds_list
