"""Augmentation de données sur GPU (batch entier, une transformation différente par image)."""
import math

import torch
import torch.nn.functional as F


def strong_augment(x, rotation=15.0, scale=(0.85, 1.15), translate=0.08,
                   brightness=0.1, contrast=(0.8, 1.2), noise=0.02):
    """x : float (N,1,H,W) dans [0,1]. Retourne un batch augmenté, toujours dans [0,1].

    - flip horizontal (p=0.5)
    - transformation affine aléatoire : rotation, zoom, translation (le fond ajouté est noir, comme le scanner)
    - luminosité / contraste aléatoires
    - léger bruit gaussien
    """
    n, dev = x.size(0), x.device

    # --- géométrie : matrice affine 2x3 par image
    ang = (torch.rand(n, device=dev) * 2 - 1) * math.radians(rotation)
    sc = torch.empty(n, device=dev).uniform_(*scale)
    flip = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    tx, ty = ((torch.rand(2, n, device=dev) * 2 - 1) * translate * 2)  # coordonnées normalisées [-1,1]
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([
        torch.stack([cos * flip, -sin, tx], 1),
        torch.stack([sin * flip, cos, ty], 1),
    ], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # --- intensité
    c = torch.empty(n, 1, 1, 1, device=dev).uniform_(*contrast)
    b = (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * brightness
    mean = x.mean((2, 3), keepdim=True)
    x = (x - mean) * c + mean + b
    if noise > 0:
        x = x + torch.randn_like(x) * noise
    return x.clamp_(0, 1)


def mixup_cutmix(x, y, alpha=0.4, cutmix_prob=0.5):
    """MixUp ou CutMix (au hasard) sur un batch. Retourne (x_mélangé, y_a, y_b, lam).

    La loss devient : lam * CE(sortie, y_a) + (1 - lam) * CE(sortie, y_b).
    - MixUp  : les deux images sont superposées en transparence (x = lam*x_a + (1-lam)*x_b)
    - CutMix : un rectangle de l'image b est collé dans l'image a ; lam = part de l'image a restante
    """
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(x.size(0), device=x.device)
    if torch.rand(1).item() < cutmix_prob:
        h, w = x.shape[-2:]
        ch, cw = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
        cy, cx = torch.randint(h, (1,)).item(), torch.randint(w, (1,)).item()
        y0, y1 = max(cy - ch // 2, 0), min(cy + ch // 2, h)
        x0, x1 = max(cx - cw // 2, 0), min(cx + cw // 2, w)
        x = x.clone()
        x[..., y0:y1, x0:x1] = x[perm][..., y0:y1, x0:x1]
        lam = 1 - (y1 - y0) * (x1 - x0) / (h * w)
    else:
        x = lam * x + (1 - lam) * x[perm]
    return x, y, y[perm], lam
