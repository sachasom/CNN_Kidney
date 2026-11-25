"""Entraîne un modèle à partir d'un config YAML et crée results/<model>-vN/.

Usage : .venv/bin/python train.py --config configs/model1.yaml

Contenu du run :
  weights/best.pt    meilleur modèle (selon train.select_metric : acc par défaut, ou f1_macro)
  weights/last.pt    dernier epoch
  train_loss.csv     epoch, loss, acc
  valid_loss.csv     epoch, loss, acc, f1_macro

Options du config (toutes facultatives, défauts = comportement du modèle 1) :
  data.augment          flip | strong | none           (défaut flip)
  train.label_smoothing                                (défaut 0)
  train.select_metric   acc | f1_macro                 (défaut acc)
  train.patience        early stopping en epochs       (défaut : pas d'arrêt anticipé)
  train.min_epochs      pas d'early stopping avant cet epoch (défaut : fin de la montée du LR OneCycle)
  train.loss            ce | focal                     (défaut ce)
  train.focal_gamma                                    (défaut 2)
  data.crop             recadrage sur le corps         (défaut false)
  train.sampler         random | balanced              (défaut random)
  train.mixup           {alpha, cutmix_prob, prob}     (défaut : désactivé)
  name                  préfixe du dossier de run      (défaut : nom du modèle)
  config.yaml        copie exacte du config utilisé
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score

from src import get_model
from src.augment import mixup_cutmix
from src.data import CLASSES, GpuLoader, load_split, prep

RESULTS_DIR = Path("results")


def new_run_dir(model_name):
    """results/model1-v1, puis -v2, -v3... (jamais d'écrasement)."""
    v = 1
    while (RESULTS_DIR / f"{model_name}-v{v}").exists():
        v += 1
    run_dir = RESULTS_DIR / f"{model_name}-v{v}"
    (run_dir / "weights").mkdir(parents=True)
    return run_dir


@torch.no_grad()
def run_epoch_eval(model, loader, criterion):
    model.eval()
    loss_sum, n, preds, targets = 0.0, 0, [], []
    for xb, yb in loader:
        # NB : pas d'autocast fp16 — sur GTX 16xx il produit des NaN (et pas de Tensor Cores)
        out = model(prep(xb, train=False))
        loss_sum += criterion(out, yb).item() * yb.size(0)
        n += yb.size(0)
        preds.append(out.argmax(1).cpu())
        targets.append(yb.cpu())
    p, t = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return loss_sum / n, (p == t).mean(), f1_score(t, p, average="macro")


def train(config_path, seed=None):
    """seed : remplace la seed du config (pour lancer la même config plusieurs fois). Le run s'appelle alors
    results/<name>-s<seed>-vN."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    run_name = cfg.get("name", cfg["model"]["name"])
    if seed is not None:
        cfg["seed"] = seed
        run_name = f"{run_name}-s{seed}"
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda"
    torch.backends.cudnn.benchmark = True

    run_dir = new_run_dir(run_name)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"Run : {run_dir}  |  GPU : {torch.cuda.get_device_name(0)}")

    tc, size = cfg["train"], cfg["data"]["img_size"]
    augment = cfg["data"].get("augment", "flip")
    select_metric = tc.get("select_metric", "acc")
    patience = tc.get("patience")
    # OneCycleLR fait monter le LR pendant les 30 premiers % : on n'arrête jamais avant la fin de cette montée
    min_epochs = tc.get("min_epochs", int(np.ceil(0.3 * tc["epochs"])))
    crop = cfg["data"].get("crop", False)
    mixup = tc.get("mixup")  # ex. {alpha: 0.4, cutmix_prob: 0.5, prob: 0.8}
    X_tr, y_tr = load_split("train", size, crop)
    X_val, y_val = load_split("valid", size, crop)
    train_dl = GpuLoader(X_tr, y_tr, tc["batch_size"], shuffle=True, device=device,
                         balanced=tc.get("sampler", "random") == "balanced")
    val_dl = GpuLoader(X_val, y_val, 256, shuffle=False, device=device)

    model = get_model(cfg["model"]["name"], n_classes=len(CLASSES), **cfg["model"].get("params", {})).to(device)
    print(f"Paramètres : {sum(p.numel() for p in model.parameters()):,}")

    weights = None
    if tc.get("class_weights"):
        counts = np.bincount(y_tr)
        weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32, device=device)
    if tc.get("loss", "ce") == "focal":
        from src.losses import FocalLoss
        criterion = FocalLoss(tc.get("focal_gamma", 2.0), weights, tc.get("label_smoothing", 0.0))
    else:
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=tc.get("label_smoothing", 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=tc["lr"], total_steps=tc["epochs"] * len(train_dl))

    train_csv = open(run_dir / "train_loss.csv", "w", newline="")
    valid_csv = open(run_dir / "valid_loss.csv", "w", newline="")
    tw, vw = csv.writer(train_csv), csv.writer(valid_csv)
    tw.writerow(["epoch", "loss", "acc"])
    vw.writerow(["epoch", "loss", "acc", "f1_macro"])

    best_score, best_epoch = -1.0, 0
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        t0, loss_sum, correct, n = time.time(), 0.0, 0, 0
        for xb, yb in train_dl:
            xb = prep(xb, train=True, augment=augment)
            optimizer.zero_grad(set_to_none=True)
            if mixup and torch.rand(1).item() < mixup.get("prob", 1.0):
                xb, ya, yb2, lam = mixup_cutmix(xb, yb, mixup.get("alpha", 0.4), mixup.get("cutmix_prob", 0.5))
                out = model(xb)
                loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb2)
            else:
                out = model(xb)
                loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.detach() * yb.size(0)
            correct += (out.argmax(1) == yb).sum()
            n += yb.size(0)
        tr_loss, tr_acc = loss_sum.item() / n, correct.item() / n
        val_loss, val_acc, val_f1 = run_epoch_eval(model, val_dl, criterion)

        tw.writerow([epoch, f"{tr_loss:.6f}", f"{tr_acc:.6f}"])
        vw.writerow([epoch, f"{val_loss:.6f}", f"{val_acc:.6f}", f"{val_f1:.6f}"])
        train_csv.flush()
        valid_csv.flush()
        print(f"Epoch {epoch:2d}/{tc['epochs']} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f} | {time.time() - t0:.1f}s")

        score = val_f1 if select_metric == "f1_macro" else val_acc
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save(model.state_dict(), run_dir / "weights" / "best.pt")
        elif patience and epoch >= min_epochs and epoch - best_epoch >= patience:
            print(f"Early stopping : pas d'amélioration du val {select_metric} depuis {patience} epochs")
            break
    torch.save(model.state_dict(), run_dir / "weights" / "last.pt")
    train_csv.close()
    valid_csv.close()
    print(f"Meilleur val {select_metric} : {best_score:.4f} (epoch {best_epoch})")
    return run_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model1.yaml")
    train(ap.parse_args().config)
