import numpy as np
import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F

from utils.weight_init import weight_init


def u3pblock(in_ch, out_ch, num_block=2, kernel_size=3, padding=1, down_sample=False):
    m = []
    if down_sample:
        m.append(nn.MaxPool2d(kernel_size=2))
    for _ in range(num_block):
        m += [nn.Conv2d(in_ch, out_ch, kernel_size, bias=False, padding=padding),
              nn.BatchNorm2d(out_ch),
              nn.ReLU(inplace=True)]
        in_ch = out_ch
    return nn.Sequential(*m)

def en2dec_layer(in_ch, out_ch, scale):
    m = [nn.Identity()] if scale == 1 else [nn.MaxPool2d(scale, scale, ceil_mode=True)]
    m.append(u3pblock(in_ch, out_ch, num_block=1))
    return nn.Sequential(*m)

def dec2dec_layer(in_ch, out_ch, scale, efficient=False):
    up = [nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True) if scale != 1 else nn.Identity()]
    m = [u3pblock(in_ch, out_ch, num_block=1)]
    efficient = True
    if efficient:
        m = m + up
    else:
        m = up + m  # used in paper
    return nn.Sequential(*m)

        
class FullScaleSkipConnect(nn.Module):
    def __init__(self, 
                 en_channels,   # encoder out channels, high to low
                 num_dec,       # number of decoder out
                 skip_ch=64, 
                 en_scales=None, 
                 dec_scales=None,
                 bottom_dec_ch=1024):

        super().__init__()
        concat_ch = skip_ch * (len(en_channels) + num_dec)

        # encoder maps to decoder maps connections
        self.en2dec_layers = nn.ModuleList()
        this_ch = en_channels[0]
        if en_scales is None:
            en_scales = []
            for ch in en_channels:
                en_scales.append(this_ch // ch)
        for ch, scale in zip(en_channels, en_scales):
            self.en2dec_layers.append(en2dec_layer(ch, skip_ch, scale))
        
        # decoder maps to decoder maps connections
        self.dec2dec_layers = nn.ModuleList()
        if dec_scales is None:
            dec_scales = []
            for ii in reversed(range(num_dec)):
                dec_scales.append(2 ** (ii + 1))
        for ii, scale in enumerate(dec_scales):
            dec_ch = bottom_dec_ch if ii == 0 else concat_ch
            self.dec2dec_layers.append(dec2dec_layer(dec_ch, skip_ch, scale))

        self.fuse_layer = u3pblock(concat_ch, concat_ch, 1)

    def forward(self, en_maps, dec_maps=None):
        out = []
        for en_map, layer in zip(en_maps, self.en2dec_layers):
            out.append(layer(en_map))
        if dec_maps is not None or len(dec_maps) > 0:
            for dec_map, layer in zip(dec_maps, self.dec2dec_layers):
                out.append(layer(dec_map))
        return self.fuse_layer(torch.cat(out, 1))


class U3PEncoderDefault(nn.Module):
    def __init__(self, channels = [3, 64, 128, 256, 512, 1024], num_block=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.downsample_list = nn.Module()
        for ii, (ch_in, ch_out) in enumerate(zip(channels[:-1], channels[1:])):
            self.layers.append(u3pblock(ch_in, ch_out, num_block, down_sample= ii > 0))
        self.channels = channels
        self.apply(weight_init)
        
    def forward(self, x):
        encoder_out = []
        for layer in self.layers:
            x = layer(x)
            encoder_out.append(x)
        return encoder_out


class U3PDecoder(nn.Module):
    def __init__(self, en_channels = [64, 128, 256, 512, 1024], skip_ch=64):
        super().__init__()
        self.decoders = nn.ModuleDict()
        en_channels = en_channels[::-1]
        for ii in range(len(en_channels)):
            if ii == 0:
                # first decoding output is identity mapping of last encoder map
                self.decoders['decoder1'] = nn.Identity()
                continue
            self.decoders[f'decoder{ii+1}'] = FullScaleSkipConnect(
                                                en_channels[ii:], 
                                                num_dec=ii, 
                                                skip_ch=skip_ch, 
                                                bottom_dec_ch=en_channels[0]
                                            )

    def forward(self, enc_map_list:List[torch.Tensor]):
        dec_map_list = []
        enc_map_list = enc_map_list[::-1]
        layer: FullScaleSkipConnect
        for ii, layer_key in enumerate(self.decoders):
            layer = self.decoders[layer_key]
            if ii == 0:
                dec_map_list.append(layer(enc_map_list[0]))
                continue
            dec_map_list.append(layer(enc_map_list[ii: ], dec_map_list))
        return dec_map_list


class UNet3Plus(nn.Module):

    def __init__(self, 
                 num_classes=1,
                 skip_ch=64,
                 aux_losses=2,
                 encoder: U3PEncoderDefault = None,
                 channels=[3, 64, 128, 256, 512, 1024],
                 use_cgm=True):
        super().__init__()

        self.encoder = U3PEncoderDefault(channels) if encoder is None else encoder
        channels = self.encoder.channels
        num_decoders = len(channels) - 1
        decoder_ch = skip_ch * num_decoders

        self.decoder = U3PDecoder(self.encoder.channels[1:], skip_ch=skip_ch)
        self.decoder.apply(weight_init)
        
        self.cls = nn.Sequential(
                    nn.Dropout(p=0.5),
                    nn.Conv2d(channels[-1], 2, 1),
                    nn.AdaptiveMaxPool2d(1),
                    nn.Sigmoid()
                ) if use_cgm and num_classes <= 2 else None
        
        self.head = nn.Conv2d(decoder_ch, num_classes, 3, padding=1)
        self.head.apply(weight_init)

        if aux_losses > 0:
            self.aux_head = nn.ModuleDict()
            layer_indices = np.arange(num_decoders - aux_losses - 1, num_decoders - 1)
            for ii in layer_indices:
                ch = decoder_ch if ii != 0 else channels[-1]
                self.aux_head.add_module(f'aux_head{ii}', nn.Conv2d(ch, num_classes, 3, padding=1))
            self.aux_head.apply(weight_init)
        else:
            self.aux_head = None

    def forward(self, x): 
        _, _, h, w = x.shape
        de_out = self.decoder(self.encoder(x))
        have_obj = 1

        pred = self.resize(self.head(de_out[-1]), h, w)
        
        if self.training:
            pred = {'final_pred': pred}
            if self.aux_head is not None:
                for ii, de in enumerate(de_out[:-1]):
                    if ii == 0:
                        if self.cls is not None:
                            pred['cls'] = self.cls(de).squeeze_()
                            have_obj = torch.argmax(pred['cls'])
                    head_key = f'aux_head{ii}'
                    if head_key in self.aux_head:
                        de = de * have_obj
                        pred[f'aux{ii}'] = self.resize(self.aux_head[head_key](de), h, w)
        return pred
    
    def resize(self, x, h, w) -> torch.Tensor:
        _, _, xh, xw = x.shape
        if xh != h or xw != w:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x