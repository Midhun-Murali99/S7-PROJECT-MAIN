import torch
import torch.nn as nn
from torchvision.models import resnet50


class TomatoResNet50(nn.Module):
    """
    ResNet-50 classifier for tomato disease recognition.

    This model is suitable when the input is a single disease image and the task is
    to predict one class from several tomato disease categories. The dataset in this
    project already stores YOLO bounding boxes, but for classification we reduce each
    sample to its primary class label.
    """

    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        super().__init__()

        # Use torchvision's ResNet-50 implementation.
        # We keep it lightweight and standard for image classification.
        self.backbone = resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
