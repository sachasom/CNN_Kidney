"""Registre des modèles : ajouter ici chaque nouveau src/modelN.py."""
from src.model1 import Model1
from src.model2 import Model2

MODELS = {
    "model1": Model1,
    "model2": Model2,
}


def get_model(name, **kwargs):
    if name not in MODELS:
        raise ValueError(f"Modèle inconnu '{name}'. Disponibles : {list(MODELS)}")
    return MODELS[name](**kwargs)
