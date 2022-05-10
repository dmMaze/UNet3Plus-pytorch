import sys, os
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))
from model import build_unet3plus
import torch

def test_build_model():
    model = build_unet3plus(num_classes=21, encoder='default', pretrained=True, skip_ch=32, use_cgm=True)
    model.train()
    print(model)
    input = torch.randn(1, 3, 320, 320)
    with torch.no_grad():
        out = model(input)
        for key in out:
            print(key, out[key].shape)

if __name__ == '__main__':
    test_build_model()
