import sys, os
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))
from model import build_unet3plus
import torch

if __name__ == '__main__':
    model = build_unet3plus(num_classes=21, encoder='resnet34', pretrained=True)
    input = torch.randn(1, 3, 320, 320)
    out = model(input)