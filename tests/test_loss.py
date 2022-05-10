
import sys, os
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))

import torch

from utils.loss import build_u3p_loss
from model import build_unet3plus

if __name__ == '__main__':
    model = build_unet3plus(num_classes=21, encoder='resnet18', pretrained=False, skip_ch=32, use_cgm=False)
    dim = 320
    input = torch.randn(2, 3, dim, dim)
    target = torch.randint(0, 21, (2, dim, dim))
    # model.eval()
    # with torch.no_grad():
    out_dict = model(input)
    criterion = build_u3p_loss(loss_type='u3p', aux_weight=0.4, )
    loss = criterion(out_dict, target)
    print(loss)