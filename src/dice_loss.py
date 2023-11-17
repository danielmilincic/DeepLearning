import torch
import torch.nn.functional as F

class DiceLoss(torch.nn.Module):
    def __init__(self, num_classes=3, smooth=1.):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, outputs, targets):
        outputs = F.softmax(outputs, dim=1)  #  model outputs logits so we want to convert to probabilities
        # controlla logits

        dice_loss = 0

        for class_idx in range(self.num_classes):
            output_class = outputs[:, class_idx, ...]
            target_class = (targets == class_idx).float()

            intersection = torch.sum(output_class * target_class)
            union = torch.sum(output_class) + torch.sum(target_class) + self.smooth

            dice_loss += 1 - (2 * intersection + self.smooth) / union

        return dice_loss / self.num_classes

"""
def _iou(pred, target, size_average = True):

    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
        IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b


class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):

        return _iou(pred, target, self.size_average)

def IOU_loss(pred,label):
    iou_loss = IOU(size_average=True)
    iou_out = iou_loss(pred, label)
    print("iou_loss:", iou_out.data.cpu().numpy())
    return iou_out
"""