"""Fonctions de perte."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017) : CE multipliée par (1 - p_vrai)^gamma.

    Une image déjà bien classée (p_vrai proche de 1) ne compte presque plus ; l'entraînement se concentre sur
    les exemples difficiles (ex. les tumeurs confondues avec Normal). gamma = 0 redonne la cross-entropy.
    Compatible avec les poids de classe et le label smoothing.
    """

    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma, self.weight, self.label_smoothing = gamma, weight, label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction="none", label_smoothing=self.label_smoothing)
        p_true = torch.softmax(logits, 1).gather(1, target[:, None]).squeeze(1)
        loss = (1 - p_true).pow(self.gamma) * ce
        if self.weight is not None:
            w = self.weight[target]
            return (loss * w).sum() / w.sum()
        return loss.mean()
