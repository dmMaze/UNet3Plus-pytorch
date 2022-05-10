import torch
import torch.nn as nn
import torch.nn.functional as F
from .focalloss import FocalLoss
from .ms_ssimloss import MS_SSIMLoss, SSIMLoss
from .piqa_ssim import SSIM
from .iouloss import IoULoss

class U3PLloss(nn.Module):

    def __init__(self, loss_type='focal', aux_weight=0.4, process_input=True):
        super().__init__()
        self.aux_weight = aux_weight
        self.focal_loss = FocalLoss(ignore_index=255, size_average=True)  
        if loss_type == 'u3p':
            self.iou_loss = IoULoss(process_input=not process_input)
            # self.ms_ssim_loss = MS_SSIMLoss(process_input=not process_input)
            self.ms_ssim_loss = SSIMLoss(process_input=not process_input)
            # self.ms_ssim_loss = SSIM()
        elif loss_type != 'focal':
            raise ValueError(f'Unknown loss type: {loss_type}')
        self.loss_type = loss_type
        self.process_input = process_input

    def forward(self, preds, targets):
        if not isinstance(preds, dict):
            preds = {'final_pred': preds}
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

    def onehot_softmax(self, pred, target: torch.Tensor, process_target=True):
        _, num_classes, h, w = pred.shape
        pred = F.softmax(pred, dim=1)
        
        if process_target:
            target = torch.clamp(target, 0, num_classes)
            target = F.one_hot(target, num_classes=num_classes+1)[..., :num_classes].permute(0, 3, 1, 2).contiguous().to(pred.dtype)
        return pred, target


    def _forward_u3p(self, preds, targets):
        r'''Full-scale Deep Supervision
        '''

        loss, loss_dict = self._forward_focal(preds, targets)
        if self.process_input:
            final_pred, targets = self.onehot_softmax(preds['final_pred'], targets)
        iou_loss = self.iou_loss(final_pred, targets)
        msssim_loss = self.ms_ssim_loss(final_pred, targets)
        loss = loss + iou_loss + msssim_loss
        loss_dict['head_iou_loss'] = iou_loss.detach().item()
        loss_dict['head_msssim_loss'] = msssim_loss.detach().item()

        num_aux, aux_iou_loss, aux_msssim_loss = 0, 0., 0.
        for key in preds:
            if 'aux' in key:
                num_aux += 1
                if self.process_input:
                    preds[key], targets = self.onehot_softmax(preds[key], targets, process_target=False)
                aux_iou_loss += self.iou_loss(preds[key], targets)
                aux_msssim_loss += self.ms_ssim_loss(preds[key], targets)
        if num_aux > 0:
            aux_iou_loss /= num_aux
            aux_msssim_loss /= num_aux
            loss_dict['aux_iou_loss'] = aux_iou_loss.detach().item()
            loss_dict['aux_msssim_loss'] = aux_msssim_loss.detach().item()
            loss += (aux_iou_loss + aux_msssim_loss) * self.aux_weight
            loss_dict['total_loss'] = loss.detach().item()
        
        return loss, loss_dict

def build_u3p_loss(loss_type='focal', aux_weight=0.4, ) -> U3PLloss:
    return U3PLloss(loss_type, aux_weight)