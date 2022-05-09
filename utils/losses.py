import torch.nn as nn
import torch.nn.functional as F
import torch 



class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class U3PLloss(nn.Module):

    def __init__(self, loss_type='focal', aux_weight=0.4):
        super().__init__()
        self.aux_weight = aux_weight
        self.focal_loss = FocalLoss(ignore_index=255, size_average=True)
        self.loss_type = loss_type

    def forward(self, preds, targets):
        if self.loss_type == 'focal':
            loss_func = self.focal_loss

        loss = loss_func(preds['final_pred'], targets)
        for key in preds:
            if 'aux' in key:
                loss += loss_func(preds['final_pred'], targets) * self.aux_weight
        return loss

def build_loss(loss_type='focal', aux_weight=0.4, ) -> U3PLloss:
    return U3PLloss(loss_type, aux_weight)