import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.crf import CRF
from .layers.attention import BertSelfAttention
from .layers.bilstm import BILSTM
from transformers import BertModel,BertPreTrainedModel

class BertCrfForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCrfForNer, self).__init__(config)
        self.bert = BertModel(config)
        self.lambda_score = 0.8

        # self.dependency_mask_attention = BertSelfAttention(config)

        #Context Aggregation Feature
        self.lambda_concat_layer = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.lambda_score_layer = nn.Linear(config.hidden_size, 1, )

        self.dropout_bert = nn.Dropout(config.hidden_dropout_prob)

        #one layer bilstm
        self.bilstm = BILSTM(config.hidden_size // 2, 1, config.hidden_size, config.hidden_dropout_prob, config.num_labels, True)
        
        #Global self-attention
        self.attention_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_ut = nn.Parameter(torch.Tensor(config.hidden_size, 1))
        self.attention_ut.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.attention_z = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.dropout_lstm = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, dependency_mask=None, labels=None):
        original_outputs =self.bert(input_ids=input_ids,attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        dependency_outputs = self.bert(input_ids=input_ids,attention_mask=dependency_mask, token_type_ids=token_type_ids)[0]


        # Context Aggregation Feature
        # outputs = torch.cat([original_outputs, dependency_outputs], axis=-1)
        #
        # lambda_ = self.lambda_concat_layer(outputs)
        # lambda_ = self.lambda_score_layer(lambda_)
        # lambda_score = torch.sigmoid(lambda_)
        #
        sequence_output = self.lambda_score * original_outputs + (1-self.lambda_score) * dependency_outputs
        # sequence_output = original_outputs
        sequence_output = self.dropout_bert(sequence_output)

        #bilstm
        sequence_bilstm_output = self.bilstm(sequence_output) #(bs, seq_len, hidden_size)
        sequence_bilstm_output = self.dropout_lstm(sequence_bilstm_output)

        #Global self-attention ui
        attention_ui = torch.tanh(self.attention_layer(sequence_bilstm_output)) # (bs, seq_len, seq_len)
        attention_alpha = torch.einsum('bmd,dn->bmn',attention_ui, self.attention_ut)  # (bs, seq_len, 1)
        attention_alpha = torch.squeeze(attention_alpha, -1) # (bs, seq_len)
        attention_alpha = F.softmax(attention_alpha)
        attention_alpha = torch.unsqueeze(attention_alpha, 2) # (bs, seq_len, 1)
        attention_output = torch.sum(attention_alpha * sequence_bilstm_output, dim=1, keepdim=True)# (bs, 1, hidden_size)
        attention_output = torch.tile(attention_output, [1, sequence_bilstm_output.shape[1], 1])# (bs, seq_len, hidden_size)
        sequence_attention_output = torch.cat([attention_output, sequence_bilstm_output], axis=-1)
        sequence_attention_output = self.attention_z(sequence_attention_output) # (bs, seq_len, hidden_size)
        sequence_attention_output = torch.tanh(sequence_attention_output)

        logits = self.classifier(sequence_attention_output) #(bs, seq_len, num_labels)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions = logits, tags=labels, mask=attention_mask)
            outputs =(-1*loss,)+outputs
        return outputs # (loss), scores

    def decode(self, logits, attention_mask, decode_mask):
        viterbi_path = self.crf.decode(logits, attention_mask)
        viterbi_tags = viterbi_path.squeeze(0).cpu().numpy()
        decode_tags = []
        for i in range(viterbi_tags.shape[0]):
            decode_tag = []
            for j in range(viterbi_tags.shape[1]):
                if decode_mask[i, j]:
                    decode_tag.append(viterbi_tags[i, j])
            decode_tags.append(decode_tag[1:-1])

        return decode_tags


