import torch
import torch.nn as nn
import torch.nn.functional as F

# iou loss in official repo
# def _iou(pred, target, size_average = True):

#     b = pred.shape[0]
#     IoU = 0.0
#     for i in range(0,b):
#         #compute the IoU of the foreground
#         Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
#         Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
#         IoU1 = Iand1/Ior1

#         #IoU loss is (1-IoU1)
#         IoU = IoU + (1-IoU1)

#     return IoU/b

def binary_iou_loss(pred, target):
    Iand = torch.sum(pred * target, dim=1)
    Ior = torch.sum(pred, dim=1) + torch.sum(target, dim=1) - Iand
    IoU = 1 - Iand.sum() / Ior.sum()
    return IoU.sum()

class IoULoss(nn.Module):
    '''
    multi-classes iou loss
    '''
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(target, num_classes=num_classes)

        total_loss = 0
        for i in range(num_classes):
            loss = binary_iou_loss(pred[:, i], target[..., i])
            total_loss += loss
        return total_loss / num_classes

if __name__ == '__main__':
    # pred = torch.randint(0, 2, (6, 5, 50, 50))
    # target = torch.randint(0, 2, (6, 5, 50, 50))
    # bil = binary_iou_loss(pred, target)
    # iou = _iou(pred, target)
    # print(bil, iou)

    pred = torch.randn((6, 21, 50, 50))
    target = torch.randint(0, 21, (6, 50, 50))
    iou = IoULoss()
    rst = iou(pred, target)
    print(rst)