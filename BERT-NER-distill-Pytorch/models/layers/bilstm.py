#encoding:utf-8
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence

class BILSTM(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_layer,
                 input_size,
                 dropout_p,
                 num_classes,
                 bi_tag):

        super(BILSTM,self).__init__()
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

        self.lstm = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = num_layer,
                            batch_first = True,
                            dropout = dropout_p,
                            bidirectional = bi_tag)

    def forward(self,inputs):
        output, _ = self.lstm(inputs)
        return output