import numpy as np
import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F
import sys, os
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))

from utils.weight_init import weight_init

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        if isinstance(act, bool):
            self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        elif isinstance(act, str):
            if act == 'leaky':
                self.act = nn.LeakyReLU(0.1, inplace=True)
            elif act == 'relu':
                self.act = nn.ReLU(inplace=True)
            else:
                self.act = None
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, act=True):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c_, c2, 3, 1, g=g, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act=True, final_act=True):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c1, c_, 1, 1, act=act)
        final_act = act if final_act else False
        self.cv3 = Conv(2 * c_, c2, 1, act=final_act)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0, act=act) for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class double_conv_up_c3(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, act=True, shrink=False, interpolate=False):
        super(double_conv_up_c3, self).__init__()
        self.interpolate = interpolate
        if shrink:
            self.conv = nn.Sequential(
            nn.Conv2d(in_ch+mid_ch, mid_ch, 1, bias=False),
            C3(mid_ch, mid_ch, act=act),
            nn.ConvTranspose2d(mid_ch, out_ch, kernel_size=4, stride = 2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
            C3(in_ch+mid_ch, mid_ch, act=act),
            nn.ConvTranspose2d(mid_ch, out_ch, kernel_size=4, stride = 2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.conv(x)


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
    # m = [C3(in_ch, out_ch)]
    efficient = True
    if efficient:
        m = m + up
    else:
        m = up + m  # used in paper
    return nn.Sequential(*m)

        
class FullScaleSkipConnect(nn.Module):
    def __init__(self, 
                 en_channels,   # encoder out channels, high to low
                 en_scales,
                 num_dec,       # number of decoder out
                 skip_ch=64, 
                 dec_scales=None,
                 max_connections=None,
                 bottom_dec_ch=1024):

        super().__init__()
        concat_ch = skip_ch * (len(en_channels) + num_dec)

        # encoder maps to decoder maps connections
        self.en2dec_layers = nn.ModuleList()
        # print(en_scales)
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
        if dec_maps is not None and len(dec_maps) > 0:
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
        num_en_ch = len(en_channels)
        for ii in range(num_en_ch):
            if ii == 0:
                # first decoding output is identity mapping of last encoder map
                self.decoders['decoder1'] = nn.Identity()
                continue

            self.decoders[f'decoder{ii+1}'] = FullScaleSkipConnect(
                                                en_channels[ii:], 
                                                en_scales=2 ** np.arange(0, num_en_ch-ii),
                                                num_dec=ii, 
                                                skip_ch=skip_ch, 
                                                bottom_dec_ch=en_channels[0]
                                            )

    def forward(self, enc_map_list:List[torch.Tensor]):
        dec_map_list = []
        enc_map_list = enc_map_list[::-1]
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
        
        # self.head = nn.Conv2d(decoder_ch, num_classes, 3, padding=1)
        # self.head.apply(weight_init)
        self.head = nn.Sequential(
            # nn.Conv2d(decoder_ch, 32, 1, bias=False),
            # double_conv_up_c3(0, 32, 32),
            # nn.Conv2d(32, num_classes, 1, bias=False),
            nn.ConvTranspose2d(decoder_ch, num_classes, kernel_size=4, stride = 2, padding=1, bias=False),
            # nn.BatchNorm2d(num_classes),
            # nn.ReLU(inplace=True),
        )
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
                        de: torch.Tensor = de * have_obj
                        pred[f'aux{ii}'] = self.resize(self.aux_head[head_key](de), h, w)
        return pred
    
    def resize(self, x, h, w) -> torch.Tensor:
        _, _, xh, xw = x.shape
        if xh != h or xw != w:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x

if __name__ == '__main__':
    input = torch.randn((2, 3, 320, 320))
    model = UNet3Plus(num_classes=7)
    out = model(input)
    # print(out)