import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from .layers.crf import CRF

from transformers import BertModel,BertPreTrainedModel
import numpy as np

class BertApc(BertPreTrainedModel):
    def __init__(self, config):

        super(BertApc, self).__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.dropout_bert = nn.Dropout(config.hidden_dropout_prob)

        #POS嵌入层
        self.POS_embeddings = nn.Embedding(config.num_POS_tag, config.hidden_size, padding_idx=config.pad_token_id)

        #自注意力层
        self.attention_W = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=False)
        self.attention_layer = nn.MultiheadAttention(embed_dim=config.hidden_size,
                                                     num_heads=1,
                                                     dropout=config.hidden_dropout_prob,
                                                     batch_first=True)
        #分类层
        self.fusion_dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.dropout_fusion = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, input_POS_ids, token_type_ids=None, attention_mask=None, ap_class_labels=None):

        #语义提取
        sequence_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        sequence_output = self.dropout_bert(sequence_output)


        #POS嵌入层
        POS_embed = self.POS_embeddings(input_POS_ids)

        POS_embed_q, POS_embed_k, POS_embed_v = torch.tensor_split(self.attention_W(POS_embed), 3, dim=-1) #(bs, sequence_len, hidden_size)

        POS_output = self.attention_layer(POS_embed_q, POS_embed_k, POS_embed_v, attn_mask=self.create_attention_mask_from_input_mask(input_ids, attention_mask))[0]  #(bs, sequence_len, hidden_size)

        #全连接层
        sequence_output = torch.concat([sequence_output, POS_output], axis=-1)
        sequence_output = self.fusion_dense(sequence_output)
        sequence_output = self.dropout_fusion(sequence_output)


        #分类层
        logits = self.classifier(sequence_output)  # (bs, sequence_len, ap_polarity_size)
        outputs = (logits,)

        # Only keep active parts of the loss
        loss_fct = CrossEntropyLoss()
        if ap_class_labels is not None:
            ap_class_labels = torch.where(ap_class_labels==-100, self.config.ap_class_map["O"], ap_class_labels)
            loss = self.crf(emissions = logits, tags=ap_class_labels, mask=attention_mask)
            outputs =(-1*loss,)+outputs
        # if ap_class_labels is not None:
        #     if attention_mask is not None:
        #         active_loss = attention_mask.reshape(-1, ) == 1
        #         active_logits = logits.view(-1, self.config.num_labels)
        #         active_labels = torch.where(
        #             active_loss, ap_class_labels.reshape(-1, ), torch.tensor(loss_fct.ignore_index).type_as(ap_class_labels)
        #         )
        #         loss = loss_fct(active_logits, active_labels)
        #     else:
        #         loss = loss_fct(logits.view(-1, self.num_labels), ap_class_labels.view(-1))
        #     outputs = (loss,) + outputs
        return outputs

    def create_attention_mask_from_input_mask(self, from_tensor, to_mask):
        """Create 3D attention mask from a 2D tensor mask.
        Args:
          from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
          to_mask: int32 Tensor of shape [batch_size, to_seq_length].
        Returns:
          float Tensor of shape [batch_size, from_seq_length, to_seq_length].
        """
        from_shape = from_tensor.shape
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]

        to_shape = to_mask.shape
        to_seq_length = to_shape[1]

        to_mask = to_mask.reshape(batch_size, 1, to_seq_length).float()

        broadcast_ones = torch.ones((batch_size, from_seq_length, 1), dtype=torch.float)

        # Here we broadcast along two dimensions to create the mask.
        mask = broadcast_ones * to_mask

        return mask
    def decode(self, logits, mask):
        preds = torch.argmax(F.log_softmax(logits, dim=2), dim=2).cpu().numpy()
        batch_size, seq_len = preds.shape
        preds_list = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            for j in range(seq_len):
                if mask[i, j]:
                    preds_list[i].append(preds[i,j])
        return preds_list

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


