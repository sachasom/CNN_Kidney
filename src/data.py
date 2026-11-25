"""Données : nettoyage, découpage train/valid/test et chargement sur GPU.

Pipeline (expliqué pas à pas dans notebooks/02_preprocessing_cleaning.ipynb) :
  1. scan_raw                  inventaire de archive/ (taille, vue, hash md5…)
  2. remove_exact_duplicates   suppression des fichiers identiques octet par octet
  3. assign_series             regroupe les coupes consécutives quasi identiques en « séries »
  4. group_split               découpage stratifié PAR SÉRIE (pas de fuite train -> test)
  5. export_splits             data/{train,valid,test}/<classe>/ (hardlinks) + data/metadata.csv
Chargement : preprocess_image (gris, padding carré, resize) -> load_split (cache .npz) -> GpuLoader.
"""
import hashlib
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

RAW_DIR = Path("archive/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone")
DATA_DIR = Path("data")
CLASSES = ["Cyst", "Normal", "Stone", "Tumor"]
SPLITS = ["train", "valid", "test"]
SERIES_THRESHOLD = 0.02  # distance moyenne max entre 2 coupes consécutives d'une même série (cf. notebook 02)
_NUM_RE = re.compile(r"\((\d+)\)")


# ---------------------------------------------------------------- 1. inventaire
def scan_raw(raw_dir=RAW_DIR):
    """Une ligne par image : chemin, classe, numéro de coupe, taille, vue, md5. Trié par (classe, numéro)."""
    rows = []
    for label, cls in enumerate(CLASSES):
        for f in (Path(raw_dir) / cls).glob("*.jpg"):
            with Image.open(f) as im:
                w, h = im.size
            rows.append({
                "path": str(f), "filename": f.name, "cls": cls, "label": label,
                "num": int(_NUM_RE.search(f.name).group(1)),
                "width": w, "height": h,
                "view": "axial" if w == h else "coronal",  # 512x512 = axial, rectangulaire = coronal
                "md5": hashlib.md5(f.read_bytes()).hexdigest(),
            })
    return pd.DataFrame(rows).sort_values(["label", "num"]).reset_index(drop=True)


def thumbnails(paths, size=32):
    """Miniatures en niveaux de gris dans [0,1], shape (N, size*size). Décodage JPEG réduit (rapide)."""
    out = np.empty((len(paths), size * size), dtype=np.float32)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            im.draft("L", (size * 4, size * 4))
            out[i] = np.asarray(im.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32).ravel()
    return out / 255.0


# ---------------------------------------------------------------- 2. doublons exacts
def remove_exact_duplicates(df):
    """Garde la 1re occurrence de chaque md5. Retourne (df_propre, df_supprimés)."""
    dup = df.duplicated("md5", keep="first")
    return df[~dup].reset_index(drop=True), df[dup].reset_index(drop=True)


# ---------------------------------------------------------------- 3. séries de coupes
def consecutive_distances(df, thumbs):
    """Distance (moyenne |différence| des miniatures) entre chaque image et la précédente de la même classe.

    NaN pour la première image de chaque classe. df doit être trié par (classe, numéro).
    """
    d = np.full(len(df), np.nan, dtype=np.float32)
    d[1:] = np.abs(thumbs[1:] - thumbs[:-1]).mean(1)
    d[(df["cls"] != df["cls"].shift()).to_numpy()] = np.nan
    return d


def assign_series(df, dist, threshold=SERIES_THRESHOLD):
    """Nouvelle série quand : nouvelle classe, changement de taille d'image, ou distance > threshold."""
    new = (
        np.isnan(dist)
        | (dist > threshold)
        | (df["width"] != df["width"].shift()).to_numpy()
        | (df["height"] != df["height"].shift()).to_numpy()
    )
    df = df.copy()
    idx = np.cumsum(new)
    df["series"] = [f"{c}_{i:04d}" for c, i in zip(df["cls"], idx)]
    return df


# ---------------------------------------------------------------- 4. split par groupe
def group_split(df, val_size=0.15, test_size=0.15, seed=42):
    """Découpage stratifié par classe, où une série entière va dans un seul split.

    Pour chaque classe : séries mélangées, puis remplissage glouton de test, valid, et le reste en train.
    """
    rng = np.random.RandomState(seed)
    df = df.copy()
    df["split"] = "train"
    for cls in CLASSES:
        sizes = df[df["cls"] == cls].groupby("series").size()
        series = sizes.index.to_numpy().copy()
        rng.shuffle(series)
        total, acc = sizes.sum(), 0
        for s in series:
            if acc < test_size * total:
                split = "test"
            elif acc < (test_size + val_size) * total:
                split = "valid"
            else:
                break
            df.loc[df["series"] == s, "split"] = split
            acc += sizes[s]
    return df


# ---------------------------------------------------------------- 5. export
def export_splits(df, data_dir=DATA_DIR, overwrite=False):
    """Crée data/{split}/<classe>/<fichier> en hardlinks (0 octet de plus) + data/metadata.csv."""
    data_dir = Path(data_dir)
    if (data_dir / "train").exists():
        if not overwrite:
            raise FileExistsError(f"{data_dir}/train existe déjà (passer overwrite=True pour refaire le split)")
        for sub in SPLITS + ["cache"]:
            shutil.rmtree(data_dir / sub, ignore_errors=True)
    for row in df.itertuples():
        dst = data_dir / row.split / row.cls / row.filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(row.path, dst)
    df.to_csv(data_dir / "metadata.csv", index=False)


def prepare_data(threshold=SERIES_THRESHOLD, val_size=0.15, test_size=0.15, seed=42, overwrite=False):
    """Pipeline complet 1 -> 5 (même code que le notebook 02). Ne fait rien si le split existe déjà."""
    if (DATA_DIR / "train").exists() and not overwrite:
        print(f"Split déjà présent dans {DATA_DIR}/ — on le réutilise.")
        return pd.read_csv(DATA_DIR / "metadata.csv")
    df = scan_raw()
    df, removed = remove_exact_duplicates(df)
    df = assign_series(df, consecutive_distances(df, thumbnails(df["path"])), threshold)
    df = group_split(df, val_size, test_size, seed)
    export_splits(df, overwrite=overwrite)
    print(f"{len(removed)} doublons supprimés, {df['series'].nunique()} séries")
    print(df.groupby(["split", "cls"]).size().unstack())
    return df


# ---------------------------------------------------------------- chargement
def body_bbox(gray, threshold=15, margin=0.03):
    """Boîte englobante du corps du patient : plus grande zone claire connexe.

    Le seuillage + ouverture morphologique sépare le corps de la table du scanner et du bruit de fond.
    Retourne (x0, y0, x1, y1) en pixels.
    """
    from scipy import ndimage
    mask = ndimage.binary_opening(gray > threshold, iterations=3)
    labels, n = ndimage.label(mask)
    if n == 0:
        return 0, 0, gray.shape[1], gray.shape[0]
    largest = np.argmax(np.bincount(labels.ravel())[1:]) + 1
    ys, xs = np.where(labels == largest)
    h, w = gray.shape
    my, mx = int(margin * h), int(margin * w)
    return max(xs.min() - mx, 0), max(ys.min() - my, 0), min(xs.max() + mx + 1, w), min(ys.max() + my + 1, h)


def preprocess_image(path, img_size, crop=False):
    """Niveaux de gris -> [recadrage sur le corps] -> padding noir pour rendre carré -> resize.

    Le padding évite de déformer les coupes coronales (rectangulaires).
    crop=True : on retire d'abord le fond noir et la table du scanner (cf. notebook 05).
    """
    with Image.open(path) as im:
        im = im.convert("L")
    if crop:
        im = im.crop(body_bbox(np.asarray(im)))
    w, h = im.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), 0)
    canvas.paste(im, ((side - w) // 2, (side - h) // 2))
    return np.asarray(canvas.resize((img_size, img_size), Image.BILINEAR), dtype=np.uint8)


def list_split_files(split, data_dir=DATA_DIR):
    """Fichiers d'un split dans l'ordre exact utilisé par load_split -> [(chemin, label)]."""
    return [(f, label) for label, cls in enumerate(CLASSES)
            for f in sorted((Path(data_dir) / split / cls).glob("*.jpg"))]


def load_split(split, img_size, crop=False, data_dir=DATA_DIR):
    """Charge un split prétraité -> (X uint8 [N,H,W], y int64). Mis en cache dans data/cache/."""
    cache = Path(data_dir) / "cache" / f"{split}_{img_size}{'_crop' if crop else ''}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["X"], d["y"]
    files = list_split_files(split, data_dir)
    X = np.stack([preprocess_image(f, img_size, crop) for f, _ in files])
    y = np.array([label for _, label in files], dtype=np.int64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, y=y)
    return X, y


class GpuLoader:
    """Tout le split tient sur le GPU en uint8 (~140 Mo en 128px) : pas de DataLoader CPU."""

    def __init__(self, X, y, batch_size, shuffle, device="cuda", balanced=False):
        """balanced=True : tirage avec remise, chaque classe a la même probabilité (sampler équilibré)."""
        self.X = torch.from_numpy(X).unsqueeze(1).to(device)  # (N,1,H,W)
        self.y = torch.from_numpy(y).to(device)
        self.batch_size, self.shuffle, self.device = batch_size, shuffle, device
        self.sample_weights = None
        if balanced:
            counts = torch.bincount(self.y).float()
            self.sample_weights = (1.0 / counts)[self.y]

    def __len__(self):
        return (len(self.X) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.X)
        if self.sample_weights is not None:
            idx = torch.multinomial(self.sample_weights, n, replacement=True)
        elif self.shuffle:
            idx = torch.randperm(n, device=self.device)
        else:
            idx = torch.arange(n, device=self.device)
        for i in range(0, n, self.batch_size):
            b = idx[i:i + self.batch_size]
            yield self.X[b], self.y[b]


def prep(xb, train, augment="flip"):
    """uint8 -> float normalisé dans [-1, 1], + augmentation en train.

    augment = "flip"   : flip horizontal aléatoire (modèle 1)
              "strong" : flip + rotation/zoom/translation + contraste/luminosité + bruit (src/augment.py)
              "none"   : aucune
    """
    xb = xb.float().div_(255.0)
    if train and augment == "flip":
        flip = torch.rand(xb.size(0), device=xb.device) < 0.5
        xb[flip] = xb[flip].flip(-1)
    elif train and augment == "strong":
        from src.augment import strong_augment
        xb = strong_augment(xb)
    return (xb - 0.5) / 0.5
