import torch
import torch.nn as nn
from torchvision.models import resnet50


class TomatoResNet50(nn.Module):
    """ResNet-50 classifier for tomato disease recognition."""

    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        super().__init__()
        self.backbone = resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
