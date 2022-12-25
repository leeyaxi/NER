import numpy as np
import torch
import torch.nn as nn


class TransE(nn.Module):

    def __init__(self, tail_count, relation_count, norm=1, dim=100, margin=1.0):
        super(TransE, self).__init__()
        self.tail_count = tail_count
        self.relation_count = relation_count
        self.norm = norm
        self.dim = dim
        self.relations_emb = self._init_relation_emb()
        self.tail_emd = self._init_tail_emb()
        self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')

    def _init_relation_emb(self):
        relations_emb = nn.Embedding(num_embeddings=self.relation_count + 1,
                                     embedding_dim=self.dim,
                                     padding_idx=0)
        # uniform_range = 6 / np.sqrt(self.dim)
        # relations_emb.weight.data.uniform_(-uniform_range, uniform_range)
        # # -1 to avoid nan for OOV vector
        # relations_emb.weight.data[:-1, :].div_(relations_emb.weight.data[:-1, :].norm(p=1, dim=1, keepdim=True))
        return relations_emb

    def _init_tail_emb(self):
        tail_emb = nn.Embedding(num_embeddings=self.tail_count + 1,
                                 embedding_dim=self.dim,
                                 padding_idx=0)
        # uniform_range = 6 / np.sqrt(self.dim)
        # tail_emb.weight.data.uniform_(-uniform_range, uniform_range)
        # # -1 to avoid nan for OOV vector
        # tail_emb.weight.data[:-1, :].div_(tail_emb.weight.data[:-1, :].norm(p=1, dim=1, keepdim=True))
        return tail_emb

    def forward(self, positive_head, positive_tail, positive_relation,
                negtive_head, negtive_tail, negtive_relation, attn_mask):
        """Return model losses based on the input.
        """
        positive_distances = self._distance(positive_head, positive_tail, positive_relation)
        negative_distances = self._distance(negtive_head, negtive_tail, negtive_relation)

        return self.loss(positive_distances, negative_distances, attn_mask), torch.squeeze(self.relations_emb(positive_relation), 2)

    def predict(self, heads, tails, relations):
        """Calculated dissimilarity score for given triplets.

        :param triplets: triplets in Bx3 shape (B - batch, 3 - head, relation and tail)
        :return: dissimilarity score for given triplets
        """
        return self._distance(heads, tails, relations)

    def loss(self, positive_distances, negative_distances, attn_mask):
        target = torch.full(size=attn_mask.shape, fill_value=-1, dtype=torch.long)
        target = target * attn_mask
        return self.criterion(positive_distances.reshape(-1, 1), negative_distances.reshape(-1, 1), target.reshape(-1, 1))

    def _distance(self, heads_emd, tails, relations):
        """Triplets should have shape Bx3 where dim 3 are head id, relation id, tail id."""
        return (heads_emd + torch.squeeze(self.relations_emb(relations), 2) - torch.squeeze(self.tail_emd(tails), 2)).norm(p=self.norm,
                                                                                                          dim=-1)
