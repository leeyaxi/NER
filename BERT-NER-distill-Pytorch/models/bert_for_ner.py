import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.crf import CRF
from .layers.cnn import IDCNN
from .layers.bilstm import BILSTM
from transformers import BertModel,BertPreTrainedModel
from typing import List, Optional
from .layers.linears import PoolerEndLogits, PoolerStartLogits
from torch.nn import CrossEntropyLoss
from losses.focal_loss import FocalLoss
from losses.label_smoothing import LabelSmoothingCrossEntropy

class BertCrfForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCrfForNer, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,labels=None):
        outputs =self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions = logits, tags=labels, mask=attention_mask)
            outputs =(-1*loss,)+outputs
        return outputs # (loss), scores

class IDCNNCrfForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(IDCNNCrfForNer, self).__init__(config)
        word_embedding_dim = config.hidden_size
        self.embedding = Embeddings(config)
        # self.embedding = nn.Embedding(config.vocab_size, word_embedding_dim, name="embeddings.word_embeddings")
        self.idcnn = IDCNN(input_size=word_embedding_dim, seq_len=128, filters=300)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(300, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,labels=None):
        embeddings = self.embedding(input_ids)
        # embeddings = self.dropout(embeddings)
        sequence_output = self.idcnn(embeddings)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions = logits, tags=labels, mask=attention_mask)
            outputs =(-1*loss,)+outputs
        return outputs # (loss), scores

class BilstmCrfForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BilstmCrfForNer, self).__init__(config)
        word_embedding_dim = config.hidden_size
        self.embedding = Embeddings(config)
        self.lstm = BILSTM(300, 2, word_embedding_dim, config.hidden_dropout_prob, config.num_labels, True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(300 * 2, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        embeddings = self.embedding(input_ids)
        # embeddings = self.dropout(embeddings)
        sequence_output = self.lstm(embeddings)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions = logits, tags=labels, mask=attention_mask)
            outputs =(-1*loss,)+outputs
        return outputs # (loss), scores


class Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        inputs_embeds = self.word_embeddings(input_ids)
        embeddings = self.LayerNorm(inputs_embeds)
        embeddings = self.dropout(embeddings)
        return embeddings

