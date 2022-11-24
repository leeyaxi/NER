import torch
import torch.nn as nn
import torch.nn.functional as F

class CELoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, input, target):
        """
        计算交叉熵损失
        :param logits: [bsz x len x dim] or [bsz x dim]
        :param labels: [bsz x len] or [bsz x dim]
        :return:
        """
        num_labels = input.size(-1)
        logits = input.view(-1, num_labels)
        target = target.view(-1, num_labels)
        loss = F.binary_cross_entropy(F.sigmoid(logits), F.sigmoid(target), reduction='mean')
        return loss


class KLLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, input, target):
        """
        计算kl散度
        :param pred_logits:[bsz x len x dim] or [bsz x dim]
        :param target_logits:[bsz x len] or [bsz x dim]
        :return:
        """
        dim = input.size(-1)
        pred_logits = input.view(-1, dim)
        target_logits = target.view(-1, dim)
        x = F.log_softmax(pred_logits, dim=-1)
        y = F.softmax(target_logits, dim=-1)
        # note:一定要是batchmean，不能是mean，两者不一样
        loss = F.kl_div(x, y, reduction='batchmean')
        return loss
