
import sys, os
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))
from model import build_unet3plus
import torch

if __name__ == '__main__':
    model = build_unet3plus(encoder='resnet34')
    input = torch.randn(1, 3, 320, 320)
    out = model(input)
