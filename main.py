"""Pipeline complet : nettoyage + split des données (si absent) -> entraînement -> évaluation test.

Usage : .venv/bin/python main.py --config configs/model1.yaml
"""
import argparse

from eval import evaluate
from src.data import prepare_data
from train import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model1.yaml")
    args = ap.parse_args()

    prepare_data()  # nettoyage + split par séries (voir notebooks/02), ne refait rien si data/ existe
    run_dir = train(args.config)
    evaluate(run_dir)
