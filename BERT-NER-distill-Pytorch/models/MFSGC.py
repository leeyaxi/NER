import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from .layers.dynamic_rnn import DynamicLSTM
from .layers.embedding import Embeddings
from .layers.SGC import GraphConvolution
from .layers.TransE import TransE
from transformers import BertModel,BertPreTrainedModel
import numpy as np

class MFSGC(BertPreTrainedModel):
    def __init__(self, config, ap_class_map):
        config.semantic_hidden_size = 300
        config.sgc_layers = 2
        config.kg_fusion_nums_heads = 3
        super(MFSGC, self).__init__(config)
        self.config = config
        self.ap_class_map = ap_class_map

        #词嵌入层
        # self.embeddings = Embeddings(config)
        self.bert = BertModel(config)
        self.bert_dropout = nn.Dropout(config.hidden_dropout_prob)
        #语义提取层
        # self.lstm = DynamicLSTM(input_size=config.hidden_size,
        #             hidden_size=config.semantic_hidden_size // 2,
        #             num_layers=1,
        #             batch_first=True,
        #             bidirectional=True)

        self.lstm = nn.LSTM(input_size=config.hidden_size,
                            hidden_size=config.semantic_hidden_size // 2,
                            num_layers=2,
                            batch_first=True,
                            dropout=config.hidden_dropout_prob,
                            bidirectional=True)
        self.lstm_dropout = nn.Dropout(config.hidden_dropout_prob)

        #SGC层
        self.gc_layers = []
        for i in range(config.sgc_layers):
            self.gc_layers.append(GraphConvolution(config.semantic_hidden_size, config.semantic_hidden_size))

        #特征融合层
        self.transE = TransE(config.tail_size, config.relation_size, dim=config.hidden_size)
        self.kg_attention_layer = nn.MultiheadAttention(embed_dim=config.semantic_hidden_size,
                                                     num_heads=config.kg_fusion_nums_heads,
                                                     dropout=config.hidden_dropout_prob,
                                                     batch_first=True)
        self.attention_K_V = nn.Linear(config.hidden_size, config.semantic_hidden_size * 2, bias=False)
        self.attention_Q = nn.Linear(config.semantic_hidden_size, config.semantic_hidden_size, bias=False)
        self.kg_attention_dense = nn.Linear(config.semantic_hidden_size, config.semantic_hidden_size, bias=False)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attention_layer_norm = nn.LayerNorm(config.semantic_hidden_size, eps=config.layer_norm_eps)
        #分类层
        self.classifier = nn.Linear(config.semantic_hidden_size, config.num_ap_polarities)
        self.init_weights()

    def forward(self, input_ids, input_multi_fusion_adjacency_matrix, input_emotion_samples,
                ap_class_labels, input_pos_weight_q, token_type_ids=None, attention_mask=None, ap_polarity_labels=None):
        # 词嵌入
        # embeddings = self.embeddings(input_ids)
        embeddings = self.bert(input_ids=input_ids,attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        embeddings = self.bert_dropout(embeddings)
        # 语义提取
        sequence_output = self.lstm(embeddings)[0]
        sequence_output = self.lstm_dropout(sequence_output)


        #特征融合
        positive_entity_pair, negtive_entity_pair = torch.tensor_split(input_emotion_samples, 2, dim=1) #拆分正负样本, 正样本中实体用于做特征融合

        # 拆分head, tail和relation
        positive_entity, positive_entity_tail, positive_entity_relation = torch.tensor_split(positive_entity_pair, (self.config.word_max_len, self.config.word_max_len+1), dim=-1) #(bs, sample_len, word_max_len + 2)

        positive_entity_embed = self.bert.embeddings.word_embeddings(positive_entity) #(bs, sample_len, word_max_len, embedding_size)

        #对entity进行avg pooling作为entity表示
        positive_entity_mask = torch.where(positive_entity==0, 0, 1)

        positive_entity_mask = positive_entity_mask.unsqueeze(-1)   # (bs, sample_len, word_max_len, 1)
        positive_entity_sum_embeddings = torch.sum(positive_entity_embed * positive_entity_mask, dim=2)  # (bs, sample_len, embedding_size)
        positive_input_entity_len = torch.sum(positive_entity_mask, 2)  # (bs, sample_len, 1)
        positive_input_entity_len = torch.clamp(positive_input_entity_len, min=1e-9, max=512)
        positive_entity_avg_embed = positive_entity_sum_embeddings / positive_input_entity_len # (bs, sample_len, embedding_size)

        #TransE的loss
        negtive_entity, negtive_entity_tail, negtive_entity_relation = torch.tensor_split(negtive_entity_pair, (self.config.word_max_len, self.config.word_max_len+1), dim=-1) #(bs, sample_len, word_max_len + 2)
        negtive_entity_embed = self.bert.embeddings.word_embeddings(negtive_entity) #(bs, sample_len, word_max_len, embedding_size)
        negtive_entity_mask = torch.where(negtive_entity==0, 0, 1)

        negtive_entity_mask = negtive_entity_mask.unsqueeze(-1)   # (bs, sample_len, word_max_len, 1)
        negtive_entity_sum_embeddings = torch.sum(negtive_entity_embed * negtive_entity_mask, dim=2)  # (bs, sample_len, embedding_size)
        negtive_input_entity_len = torch.sum(negtive_entity_mask, 2)  # (bs, sample_len, 1)
        negtive_input_entity_len = torch.clamp(negtive_input_entity_len, min=1e-9, max=512)
        negtive_entity_avg_embed = negtive_entity_sum_embeddings / negtive_input_entity_len # (bs, sample_len, embedding_size)

        sample_attn_mask = torch.sum(positive_entity_pair, dim=-1) #(bs, sample_len)
        sample_attn_mask = torch.where(sample_attn_mask==0, 0, 1)

        transE_loss, K_r = self.transE(positive_entity_avg_embed, positive_entity_tail, positive_entity_relation,
                                       negtive_entity_avg_embed, negtive_entity_tail, negtive_entity_relation, sample_attn_mask)
        transE_loss = transE_loss.mean()


        #情感知识融合层
        sequence_sample_attn_mask = sample_attn_mask.unsqueeze(1) #(bs, 1, sample_len)
        sequence_sample_attn_mask = torch.tile(sequence_sample_attn_mask, (1, sequence_output.shape[1], 1))  #(bs, sequence_len, sample_len)

        sequence_output_q = self.attention_Q(sequence_output)
        entity_avg_embed_k_v = self.attention_K_V(K_r)
        entity_avg_embed_k, entity_avg_embed_v = torch.tensor_split(entity_avg_embed_k_v, 2, dim=-1)

        sequence_sample_attn_mask = torch.unsqueeze(sequence_sample_attn_mask, 1) #(bs, 1, sequence_len, sample_len)
        sequence_sample_attn_mask = torch.tile(sequence_sample_attn_mask, (1, self.config.kg_fusion_nums_heads, 1, 1)) #(bs, head, sequence_len, sample_len)
        sequence_sample_attn_mask = sequence_sample_attn_mask.reshape(sequence_sample_attn_mask.shape[0] * sequence_sample_attn_mask.shape[1], sequence_sample_attn_mask.shape[2], sequence_sample_attn_mask.shape[3])
        sequence_attn_output = self.kg_attention_layer(sequence_output_q, entity_avg_embed_k, entity_avg_embed_v, attn_mask=sequence_sample_attn_mask.float())[0] #(bs, sequence_len, hidden_size)
        sequence_attn_output = self.attention_dropout(self.kg_attention_dense(sequence_attn_output))
        sequence_attn_output = self.attention_dropout(sequence_attn_output)
        sequence_output = self.attention_layer_norm(sequence_attn_output + sequence_output)

        #位置权重
        input_pos_weight_q = torch.unsqueeze(input_pos_weight_q, 2) #(bs, sequence_len, 1)

        #图卷积网络层
        sgc_output = sequence_output
        for i in range(self.config.sgc_layers):
            sgc_output = input_pos_weight_q * sgc_output
            sgc_output = self.gc_layers[i](sgc_output, input_multi_fusion_adjacency_matrix)
            sgc_output = F.relu(sgc_output)

        #掩码机制层
        ap_class_mask = torch.where(ap_class_labels==self.ap_class_map["O"], 0, 1) * \
                        torch.where(ap_class_labels==-100, 0, 1)#方面词提取

        ap_class_mask = torch.unsqueeze(ap_class_mask, 2) #(bs, sequence_len, 1)
        sgc_output = sgc_output * ap_class_mask   #(bs, sequence_len, hidden_size)

        #注意力层
        alpha_mat = torch.matmul(sgc_output, sequence_output.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        output = torch.matmul(alpha, sequence_output).squeeze(1)  #(bs, hidden_size)

        #分类层
        logits = self.classifier(output)  # (bs, sequence_len, ap_polarity_size)
        outputs = (logits,)

        # Only keep active parts of the loss
        loss_fct = CrossEntropyLoss()
        if ap_polarity_labels is not None:
            loss = loss_fct(logits, ap_polarity_labels)
            # print ("polarity loss is: {}".format(loss))
            # print("transE loss is: {}".format(transE_loss))
            # outputs = (loss,) + outputs
            outputs = (loss+transE_loss,) + outputs
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


