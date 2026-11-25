"""Model 1 — CNN entraîné from scratch.

Deux variantes, choisies par le paramètre `block` du config :
- "plain"    (défaut, baseline) : 4 blocs Conv3x3 -> BN -> ReLU -> MaxPool
- "residual" (notebook 05)      : blocs résiduels à 2 convolutions + Dropout2d, comme un petit ResNet
                                  (mais SANS poids pré-entraînés)
"""
import torch.nn as nn


def plain_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ResidualBlock(nn.Module):
    """Conv-BN-ReLU-Conv-BN + raccourci (identité ou conv 1x1), puis MaxPool. Le raccourci facilite
    l'entraînement de réseaux plus profonds : le gradient passe directement par l'addition."""

    def __init__(self, cin, cout, drop2d=0.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Dropout2d(drop2d),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
        )
        self.shortcut = nn.Identity() if cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.out = nn.Sequential(nn.ReLU(inplace=True), nn.MaxPool2d(2))

    def forward(self, x):
        return self.out(self.body(x) + self.shortcut(x))


class Model1(nn.Module):
    def __init__(self, n_classes=4, in_channels=1, dropout=0.3,
                 block="plain", widths=(32, 64, 128, 256), drop2d=0.0):
        super().__init__()
        chans = [in_channels, *widths]
        if block == "plain":
            blocks = [plain_block(a, b) for a, b in zip(chans[:-1], chans[1:])]
        elif block == "residual":
            stem = nn.Sequential(nn.Conv2d(in_channels, widths[0], 3, padding=1, bias=False),
                                 nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
            chans[0] = widths[0]
            blocks = [stem] + [ResidualBlock(a, b, drop2d) for a, b in zip(chans[:-1], chans[1:])]
        else:
            raise ValueError(f"block inconnu : {block}")
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(widths[-1], n_classes))

    def forward(self, x):
        return self.head(self.features(x))
