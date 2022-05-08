import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, ResNet
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from .unet3plus import Unet3Plus, U3PEncoderDefault, U3PDecoder


resenet_cfg = {
    'resnet18': {
        'return_nodes': {
            'relu': 'layer0',
            'layer1': 'layer1',
            'layer2': 'layer2',
            'layer3': 'layer3',
            'layer4': 'layer4',
        },
        'fe_channels': [64, 64, 128, 256, 512],
        'channels': [32, 64, 128, 256, 512],
    },
    'resnet34': {
        'return_nodes': {
            'relu': 'layer0',
            'layer1': 'layer1',
            'layer2': 'layer2',
            'layer3': 'layer3',
            'layer4': 'layer4',
        },
        'fe_channels': [64, 64, 128, 256, 512],
        'channels': [32, 64, 128, 256, 512],
    }
}


class U3PResNetEncoder(nn.Module):
    '''
    ResNet encoder wrapper 
    '''
    def __init__(self, backbone='resnet18', pretrained=False) -> None:
        super().__init__()
        resnet: ResNet = globals()[backbone](pretrained=pretrained)
        cfg = resenet_cfg[backbone]
        self.backbone = create_feature_extractor(resnet, return_nodes=cfg['return_nodes'])

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
            out[f'layer{ii}'] = compress(out[f'layer{ii}'])
        out = [v for _, v in out.items()]
        return out


def build_unet3plus(encoder='default', use_cgm=False) -> Unet3Plus:
    if encoder == 'default':
        encoder = None
    elif encoder in resenet_cfg:
        encoder = U3PResNetEncoder(backbone=encoder)
    model = Unet3Plus(encoder=encoder, use_cgm=use_cgm)
    return model

