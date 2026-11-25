"""Graphiques partagés par les notebooks de modèles et de comparaison."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Palette catégorielle (ordre fixe : une couleur = une entité, jamais réattribuée)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CLASS_COLORS = {"Cyst": SERIES[0], "Normal": SERIES[1], "Stone": SERIES[2], "Tumor": SERIES[3]}
SPLIT_COLORS = {"train": SERIES[0], "valid": SERIES[1]}
# Rampe séquentielle bleue (clair -> foncé) pour les heatmaps
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f7f9fc", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
INK, INK_MUTED, GRID, AXIS = "#1a1a19", "#6b6a63", "#e1e0d9", "#c3c2b7"


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 110, "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": AXIS, "axes.labelcolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "text.color": INK,
        "lines.linewidth": 2, "lines.markersize": 5, "legend.frameon": False,
    })


def read_run(run_dir):
    """Charge les CSV de loss d'un run."""
    run_dir = Path(run_dir)
    return pd.read_csv(run_dir / "train_loss.csv"), pd.read_csv(run_dir / "valid_loss.csv")


def plot_curves(run_dir):
    """Loss et accuracy par epoch, train vs valid."""
    tr, va = read_run(run_dir)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, metric, title in [(axes[0], "loss", "Loss"), (axes[1], "acc", "Accuracy")]:
        for df, name in [(tr, "train"), (va, "valid")]:
            ax.plot(df["epoch"], df[metric], marker="o", color=SPLIT_COLORS[name], label=name)
        ax.set_title(f"{title} — {Path(run_dir).name}")
        ax.set_xlabel("epoch")
        ax.legend()
    best = va.loc[va["acc"].idxmax()]
    axes[1].annotate(f"meilleure val acc {best['acc']:.3f}\n(epoch {int(best['epoch'])})",
                     (best["epoch"], best["acc"]), xytext=(10, -35), textcoords="offset points",
                     fontsize=9, color=INK_MUTED, arrowprops=dict(arrowstyle="-", color=AXIS))
    fig.tight_layout()
    return fig


def plot_confusion(matrix, labels, title="Matrice de confusion", normalize=True):
    """Heatmap : lignes = vraie classe, colonnes = prédiction. Valeurs = % de la ligne (rappel)."""
    m = np.asarray(matrix, dtype=float)
    shown = m / m.sum(1, keepdims=True) if normalize else m
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(shown, cmap=BLUES, vmin=0, vmax=1 if normalize else None)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            txt = f"{shown[i, j]:.1%}\n({int(m[i, j])})" if normalize else f"{int(m[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if shown[i, j] > 0.55 * shown.max() else INK)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("prédit")
    ax.set_ylabel("vrai")
    ax.set_title(title)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return fig


def collect_runs(results_dir="results"):
    """Un tableau avec une ligne par run évalué (results/*/test_output.json)."""
    rows = []
    for out in sorted(Path(results_dir).glob("*/test_output.json")):
        o = json.loads(out.read_text())
        row = {"run": o["run"], "accuracy": o["accuracy"], "f1_macro": o["f1_macro"]}
        row.update({f"recall_{c}": v["recall"] for c, v in o["per_class"].items()})
        rows.append(row)
    return pd.DataFrame(rows).set_index("run") if rows else pd.DataFrame()
