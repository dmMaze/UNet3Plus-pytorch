import torch
import torch.nn.functional as F
import torch.nn as nn
from math import exp

relu = nn.ReLU(inplace=True)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def gaussian_kernel(kernel_size, sigma):
    gauss = torch.arange(0, kernel_size) - kernel_size // 2
    gauss = torch.exp(-gauss**2 / (2*sigma**2))
    return gauss / gauss.sum()

def gaussian_kernel2d(kernel_size, channel=1) -> torch.Tensor:
    '''
    2d gauss kernel, out put shape: [channel, 1, window_size, window_size]
    '''
    k = gaussian_kernel(kernel_size, 1.5)
    k = torch.einsum('i,j->ij', [k, k])
    return k.expand(channel, 1, kernel_size, kernel_size).contiguous()

def ssim_index(img1: torch.Tensor, 
               img2: torch.Tensor, 
               kernel: torch.Tensor,
               nonnegative: bool = True,
               val_range=1):
    assert img1.shape == img2.shape
    if len(img1.shape) > 3:
        channel = img1.shape[1]
    else:
        img1 = img1.unsqueeze(1)
        img2 = img2.unsqueeze(1)
        channel = 1
    if img1.dtype == torch.long:
        img1 = img1.float()
    if img2.dtype == torch.long:
        img2 = img2.float()
    L = val_range
    padding = 0
    mean1 = F.conv2d(img1, kernel, padding=padding, groups=channel)
    mean2 = F.conv2d(img2, kernel, padding=padding, groups=channel)
    mean12 = mean1 * mean2
    mean1.pow_(2)
    mean2.pow_(2)
    
    # https://en.wikipedia.org/wiki/Variance#Definition
    var1 = F.conv2d(img1 ** 2, kernel, padding=padding, groups=channel) - mean1
    var2 = F.conv2d(img2 ** 2, kernel, padding=padding, groups=channel) - mean2

    # https://en.wikipedia.org/wiki/Covariance#Definition
    covar = F.conv2d(img1 * img2, kernel, padding=padding, groups=channel) - mean12

    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2

    # https://en.wikipedia.org/wiki/Structural_similarity#Algorithm
    ssim = (2*mean12 + c1) * (2*covar + c2) \
        / ((mean1 + mean2 + c1) * (var1 + var2 + c2))
    if nonnegative:
        ssim = relu(ssim)
    # print(mean12, covar, mean1, mean2, var1, var2)
    return ssim.mean()


class SSIMLoss(nn.Module):
    r""" Multi-Class SIMM Loss for segmentation

    Args:
        win_size: (int, optional): the size of gauss kernel
        nonnegative (bool, optional): force the ssim response to be nonnegative using relu.

    Shape:
        - Input (Tensor): :math:`(B, num_classes, H, W)`, predicted probablity maps
        - Target (Tensor): :math:`(B, H, W)`, range from 0 to num_classes - 1
    """
    def __init__(self, win_size=11, nonnegative=True):

        super(SSIMLoss, self).__init__()
        self.kernel = gaussian_kernel2d(win_size, 1)
        self.win_size = win_size
        self.nonnegative = nonnegative

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        _, h, w = target.shape
        win_size = min(h, w, self.win_size)
        kernel = self.kernel if win_size == self.win_size else gaussian_kernel2d(win_size, 1)
        if kernel.device != pred.device:
            kernel.to(pred.device)
        
        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(target, num_classes=num_classes)
        
        loss = 0
        for i in range(num_classes):
            loss += 1 - ssim_index(pred[:, [i]], target[..., [i]].permute(0, 3, 1, 2), kernel, nonnegative=self.nonnegative)
        return loss / num_classes

if __name__ == '__main__':
    # gaussian(7, 1.5)
    # w0 = create_window(11, 3)
    # w = gaussian_kernel2d(11, 3)
    # print(w0[..., 0, 0], w[..., 0, 0], w0[..., 0, 0] == w[..., 0, 0])
    
    # img1 = torch.randint(0, 2, (2, 3, 100, 100), dtype=torch.float)
    # img2 = torch.randint(0, 2, (2, 3, 100, 100), dtype=torch.float)
    # # ssim
    # ssim1 = ssim(img1, img2)
    # ssim2 = ssim_index(img1, img2, nonnegative=False, kernel=w)
    # print(ssim1, ssim2)

    pred = torch.randn((6, 21, 50, 50))
    target = torch.randint(0, 21, (6, 50, 50))
    criterion = SSIMLoss()
    rst = criterion(pred, target)
    print(rst)