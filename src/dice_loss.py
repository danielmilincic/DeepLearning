import torch
import torch.nn.functional as F

class DiceLoss(torch.nn.Module):
    def __init__(self, num_classes=3, smooth=1.):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, outputs, targets):
        # Convert logits to probabilities
        outputs = F.softmax(outputs, dim=1)

        dice_loss = 0
        for class_idx in range(self.num_classes):
            output_class = outputs[:, class_idx, ...]
            target_class = (targets == class_idx).float()

            intersection = torch.sum(output_class * target_class)
            union = torch.sum(output_class) + torch.sum(target_class)

            dice_loss += 1 - (2 * intersection + self.smooth) / (union + self.smooth)

        return dice_loss / self.num_classes
