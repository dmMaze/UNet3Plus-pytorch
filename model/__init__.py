import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, ResNet
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from utils.weight_init import weight_init
from .unet3plus import UNet3Plus


resnets = ['resnet18', 'resnet34', 'resnet50', 'resnet101']
resnet_cfg = {
    'return_nodes': {
        # 'relu': 'layer0',
        'layer1': 'layer1',
        'layer2': 'layer2',
        'layer3': 'layer3',
        'layer4': 'layer4',
    },
    'resnet18': {
        'fe_channels': [64, 128, 256, 512],
        'channels': [64, 128, 256, 512],
    },
    'resnet50': {
        'fe_channels': [64, 256, 512, 1024, 2048],
        'channels': [64, 128, 256, 512, 1024],
    }
}


class U3PResNetEncoder(nn.Module):
    '''
    ResNet encoder wrapper 
    '''
    def __init__(self, backbone='resnet18', pretrained=False) -> None:
        super().__init__()
        resnet: ResNet = globals()[backbone](pretrained=pretrained)
        cfg = resnet_cfg['resnet18'] if backbone in ['resnet18', 'resnet34'] else resnet_cfg['resnet50']
        if not pretrained:
            resnet.apply(weight_init)
        self.backbone = create_feature_extractor(resnet, return_nodes=resnet_cfg['return_nodes'])

        # print(resnet)
        # input = torch.randn(1, 3, 320, 320)
        # out = self.backbone(input)

        self.compress_convs = nn.ModuleList()
        for ii, (fe_ch, ch) in enumerate(zip(cfg['fe_channels'], cfg['channels'])):
            if fe_ch != ch:
                self.compress_convs.append(nn.Conv2d(fe_ch, ch, 1, bias=False))
            else:
                self.compress_convs.append(nn.Identity())
        self.channels = [3] + cfg['channels']
        
    def forward(self, x):
        out = self.backbone(x)
        for ii, compress in enumerate(self.compress_convs):
            out[f'layer{ii+1}'] = compress(out[f'layer{ii+1}'])
        out = [v for _, v in out.items()]
        return out


def build_unet3plus(num_classes, encoder='default', skip_ch=64, aux_losses=2, use_cgm=False, pretrained=False, dropout=0.3) -> UNet3Plus:
    if encoder == 'default':
        encoder = None
    elif encoder in resnets:
        encoder = U3PResNetEncoder(backbone=encoder, pretrained=pretrained)
    else:
        raise ValueError(f'Unsupported backbone : {encoder}')
    model = UNet3Plus(num_classes, skip_ch, aux_losses, encoder, use_cgm=use_cgm, dropout=dropout)
    return model

