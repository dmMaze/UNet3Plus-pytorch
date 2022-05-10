import torch.nn as nn
from .focalloss import FocalLoss
from .ms_ssimloss import MS_SSIMLoss, SSIMLoss
from .iouloss import IoULoss

class U3PLloss(nn.Module):

    def __init__(self, loss_type='focal', aux_weight=0.4):
        super().__init__()
        self.aux_weight = aux_weight
        self.focal_loss = FocalLoss(ignore_index=255, size_average=True)  
        if loss_type == 'u3p':
            self.iou_loss = IoULoss()
            self.ms_ssim_loss = MS_SSIMLoss()
        else:
            raise ValueError(f'Unknown loss type: {loss_type}')
        self.loss_type = loss_type

    def forward(self, preds, targets):
        if self.loss_type == 'focal':
            return self._forward_focal(preds, targets)
        elif self.loss_type == 'u3p':
            return self._forward_u3p(preds, targets)

    def _forward_focal(self, preds, targets):
        loss_dict = {}
        loss = self.focal_loss(preds['final_pred'], targets)
        loss_dict['head_focal_loss'] = loss.detach().item()     # for logging
        num_aux, aux_loss = 0, 0.

        for key in preds:
            if 'aux' in key:
                num_aux += 1
                aux_loss += self.focal_loss(preds[key], targets)
        if num_aux > 0:
            aux_loss = aux_loss / num_aux * self.aux_weight
            loss_dict['aux_focal_loss'] = aux_loss.detach().item()
            loss += aux_loss
            loss_dict['total_loss'] = loss.detach().item()
        
        return loss, loss_dict

    def _forward_u3p(self, preds, targets):
        r'''Full-scale Deep Supervision
        '''
        loss, loss_dict = self._forward_focal(preds, targets)
        iou_loss = self.iou_loss(preds['final_pred'], targets)
        msssim_loss = self.ms_ssim_loss(preds['final_pred'], targets)
        loss = loss + iou_loss + msssim_loss
        loss_dict['head_iou_loss'] = iou_loss.detach().item()
        loss_dict['head_msssim_loss'] = msssim_loss.detach().item()

        num_aux, aux_iou_loss, aux_msssim_loss = 0, 0., 0.
        for key in preds:
            if 'aux' in key:
                num_aux += 1
                aux_iou_loss += self.iou_loss(preds[key], targets)
                aux_msssim_loss += self.ms_ssim_loss(preds[key], targets)
        if num_aux > 0:
            loss_dict['aux_iou_loss'] = aux_iou_loss.detach().item()
            loss += (aux_iou_loss + aux_msssim_loss) * self.aux_weight / num_aux
            loss_dict['total_loss'] = loss.detach().item()
        
        return loss, loss_dict

def build_loss(loss_type='focal', aux_weight=0.4, ) -> U3PLloss:
    return U3PLloss(loss_type, aux_weight)