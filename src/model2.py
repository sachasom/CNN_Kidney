"""Model 2 — transfer learning : ResNet18 pré-entraîné sur ImageNet, fine-tuné en entier."""
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Model2(nn.Module):
    def __init__(self, n_classes=4, pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, n_classes))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        # Entrée du pipeline : 1 canal dans [-1, 1]. ResNet attend 3 canaux normalisés « ImageNet ».
        x = (x + 1) / 2
        x = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        return self.backbone(x)
