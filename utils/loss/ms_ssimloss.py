import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

CUDA_LAUNCH_BLOCKING=1
USE_JIT = False
relu = nn.ReLU(inplace=True)

if USE_JIT:
    _jit = torch.jit.script
else:
    _jit = lambda f: f

@_jit
def gaussian_kernel(kernel_size: int, sigma: float):
    gauss = torch.arange(0, kernel_size) - kernel_size // 2
    gauss = torch.exp(-gauss**2 / (2*sigma**2))
    return gauss / gauss.sum()

@_jit
def gaussian_kernel2d(kernel_size: int, channel: int = 1) -> Tensor:
    '''
    2d gauss kernel, out put shape: [channel, 1, window_size, window_size]
    '''
    k = gaussian_kernel(kernel_size, 1.5)
    k = torch.einsum('i,j->ij', [k, k])
    return k.expand(channel, 1, kernel_size, kernel_size).contiguous()

@_jit
def ssim_index(img1: Tensor, 
               img2: Tensor, 
               kernel: Tensor,
               nonnegative: bool = True,
               channel_avg: bool = False,
               val_range: float = 1.):
    assert img1.shape == img2.shape
    if len(img1.shape) > 3:
        channel = img1.shape[1]
    else:
        img1 = img1.unsqueeze(1)
        img2 = img2.unsqueeze(1)
        channel = 1
    _, channel, height, width = img1.shape
    if img1.dtype == torch.long:
        img1 = img1.float()
    if img2.dtype == torch.long:
        img2 = img2.float()
    L = val_range

    mean1 = F.conv2d(img1, kernel, padding=0, groups=channel)
    mean2 = F.conv2d(img2, kernel, padding=0, groups=channel)
    mean12 = mean1 * mean2
    mean1.pow_(2)
    mean2.pow_(2)
    
    # https://en.wikipedia.org/wiki/Variance#Definition
    var1 = F.conv2d(img1 ** 2, kernel, padding=0, groups=channel) - mean1
    var2 = F.conv2d(img2 ** 2, kernel, padding=0, groups=channel) - mean2

    # https://en.wikipedia.org/wiki/Covariance#Definition
    covar = F.conv2d(img1 * img2, kernel, padding=0, groups=channel) - mean12

    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2

    # https://en.wikipedia.org/wiki/Structural_similarity#Algorithm
    cs = (2. * covar + c2) / (var1 + var2 + c2)
    # print(covar.mean(), var1.mean(), var2.mean(), cs.mean())  # sparse input could result in large cs
    ss = (2. * mean12 + c1) / (mean1 + mean2 + c1) * cs

    if channel_avg:
        ss, cs = ss.flatten(1), cs.flatten(1)
    else:
        ss, cs = ss.flatten(2), cs.flatten(2)

    ss, cs = ss.mean(dim=-1), cs.mean(dim=-1)
    if nonnegative:
        ss, cs = relu(ss), relu(cs)
    return ss, cs

@_jit
def ms_ssim(
    x: Tensor,
    y: Tensor,
    kernel: Tensor,
    weights: Tensor,
    val_range: float = 1.,
    nonnegative: bool = True
) -> Tensor:
    r"""Returns the MS-SSIM between :math:`x` and :math:`y`.
    
    modified from https://github.com/francois-rozet/piqa/blob/master/piqa/ssim.py
    """

    css = []
    kernel_size = kernel.shape[-1]
    m = weights.numel()
    
    for i in range(m):
        if i > 0:
            x = F.avg_pool2d(x, kernel_size=2, ceil_mode=True)
            y = F.avg_pool2d(y, kernel_size=2, ceil_mode=True)
            h, w = x.shape[-2:]
            if h < kernel_size or w < kernel_size:
                weights = weights[:i] / torch.sum(weights[:i])
                break

        ss, cs = ssim_index(
            x, y, kernel,
            channel_avg=False,
            val_range=val_range,
            nonnegative=nonnegative
        )

        css.append(cs if i + 1 < m else ss)

    msss = torch.stack(css, dim=-1) ** weights
    msss = msss.prod(dim=-1).mean(dim=-1)

    return msss


class SSIMLoss(nn.Module):
    r""" Multi label SIMM Loss for segmentation

    Args:
        win_size: (int, optional): the size of gauss kernel
        nonnegative (bool, optional): force the ssim response to be nonnegative using relu.

    Shape:
        - Input (Tensor): :math:`(B, num_classes, H, W)`, predicted probablity maps
        - Target (Tensor): :math:`(B, H, W)`, range from 0 to num_classes - 1
    """
    def __init__(self, win_size: int = 11, nonnegative: bool = True):

        super(SSIMLoss, self).__init__()
        self.kernel = gaussian_kernel2d(win_size, 1)
        self.win_size = win_size
        self.nonnegative = nonnegative

    def forward(self, pred: Tensor, target: Tensor):
        _, h, w = target.shape
        win_size = min(h, w, self.win_size)
        kernel = self.kernel if win_size == self.win_size else gaussian_kernel2d(win_size, 1)
        if kernel.device != pred.device:
            kernel.to(pred.device)
        
        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        loss = 0.
        for i in range(num_classes):
            ss, _ = ssim_index(pred[:, [i]], target[:, [i]], kernel, nonnegative=self.nonnegative)
            loss += 1. - ss.mean()
        return loss / num_classes


class MS_SSIMLoss(nn.Module):
    r""" Multi label SIMM Loss for segmentation
     """
    def __init__(self, 
                 win_size: int = 11, 
                 weights: Tensor = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]), 
                 nonnegative: bool = True,
                 process_input: bool = True):

        super(MS_SSIMLoss, self).__init__()
        self.kernel = gaussian_kernel2d(win_size, 1).cuda()
        self.weights = weights.half()
        self.win_size = win_size
        self.nonnegative = nonnegative
        self.process_input = process_input

    def forward(self, pred: Tensor, target: Tensor):
        _, num_classes, h, w = pred.shape
        win_size = min(h, w, self.win_size)
        kernel = self.kernel if win_size == self.win_size else gaussian_kernel2d(win_size, 1)
        # if kernel.device != pred.device:
        #     kernel.to(pred.device)
        
        if self.process_input:
            pred = F.softmax(pred, dim=1)
            target = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        loss = 0.
        for i in range(num_classes):
            ss = ms_ssim(pred[:, [i]], target[:, [i]], kernel, self.weights, nonnegative=self.nonnegative)
            loss += 1. - ss.mean()
        return loss / num_classes

if __name__ == '__main__':
    
    pred = torch.randn((6, 21, 256, 256))
    target = torch.randint(0, 21, (6, 256, 256))
    criterion = SSIMLoss()
    rst = criterion(pred, target)
    print(rst)

    criterion = MS_SSIMLoss()
    rst = criterion(pred, target)
    print(rst)