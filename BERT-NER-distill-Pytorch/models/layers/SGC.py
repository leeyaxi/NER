import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output
class SGC(nn.Module):
    def __init__(self, n_layers, n_feat, enable_bias=True):
        super(SGC, self).__init__()
        self.n_layers = n_layers
        self.graph_aug_linear = []
        for i in range(n_layers):
            self.graph_aug_linear.append(nn.Linear(in_features=n_feat, out_features=n_feat, bias=enable_bias))

    def forward(self, x, A):
        """
        (Aij * W * x) / (dij + 1) + bias
        x (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
        A (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length)`, *optional*):
        """
        h_output = []
        g = x
        for i in range(self.n_layers):
            h = self.graph_aug_linear[i](g)
            h =  torch.einsum('bmm,bmd->bmd',A, g)
            g = h
            h_output.append(h)
        return h_output