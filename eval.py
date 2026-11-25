"""Évalue un run sur le split test et écrit results/<run>/test_output.json.

Usage : .venv/bin/python eval.py results/model1-v1 [--weights best.pt]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src import get_model
from src.data import CLASSES, GpuLoader, load_split, prep


def load_model(run_dir, weights="best.pt", device="cuda"):
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    model = get_model(cfg["model"]["name"], n_classes=len(CLASSES), **cfg["model"].get("params", {})).to(device)
    model.load_state_dict(torch.load(run_dir / "weights" / weights, map_location=device))
    return model.eval(), cfg


@torch.no_grad()
def predict(run_dir, weights="best.pt", split="test", device="cuda", tta=False):
    """-> (y_vrai, y_prédit, probabilités) dans l'ordre de src.data.list_split_files(split).

    tta=True : test-time augmentation, moyenne des probabilités sur l'image et sa version retournée.
    """
    model, cfg = load_model(run_dir, weights, device)
    X, y = load_split(split, cfg["data"]["img_size"], cfg["data"].get("crop", False))
    probs = []
    for xb, _ in GpuLoader(X, y, 256, shuffle=False, device=device):
        # NB : pas d'autocast fp16 — sur GTX 16xx il produit des NaN
        x = prep(xb, train=False)
        pr = torch.softmax(model(x), 1)
        if tta:
            pr = (pr + torch.softmax(model(x.flip(-1)), 1)) / 2
        probs.append(pr.cpu())
    probs = torch.cat(probs).numpy()
    return y, probs.argmax(1), probs


def evaluate(run_dir, weights="best.pt", split="test", verbose=True):
    run_dir = Path(run_dir)
    y, p, _ = predict(run_dir, weights, split)

    report = classification_report(y, p, target_names=CLASSES, digits=4, output_dict=True)
    output = {
        "run": run_dir.name,
        "weights": weights,
        "split": split,
        "n_samples": int(len(y)),
        "accuracy": accuracy_score(y, p),
        "f1_macro": f1_score(y, p, average="macro"),
        "per_class": {c: {k: report[c][k] for k in ["precision", "recall", "f1-score", "support"]}
                      for c in CLASSES},
        "confusion_matrix": {"labels": CLASSES, "matrix": confusion_matrix(y, p).tolist()},
    }
    out_path = run_dir / f"{split}_output.json"
    out_path.write_text(json.dumps(output, indent=2))

    if not verbose:
        return output
    print(f"=== {split.upper()} — {run_dir.name} ({weights}) ===")
    print(classification_report(y, p, target_names=CLASSES, digits=4))
    print("Matrice de confusion (lignes = vrai, colonnes = prédit) :")
    print(np.array(output["confusion_matrix"]["matrix"]))
    print(f"-> {out_path}")
    return output


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--weights", default="best.pt")
    args = ap.parse_args()
    evaluate(args.run_dir, args.weights)
