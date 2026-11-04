"""Carregamento do modelo salvo e predição de churn."""

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.config import MODEL_PATH
from src.preprocessing import clean_data


@lru_cache(maxsize=1)
def load_model(path: Path = MODEL_PATH) -> dict:
    """Carrega o artefato do modelo (pipeline + limiar) uma única vez."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {path}. Rode 'python -m src.train' primeiro."
        )
    return joblib.load(path)


def risk_level(probability: float) -> str:
    """Traduz a probabilidade em uma faixa de risco para o time de negócio."""
    if probability >= 0.7:
        return "alto"
    if probability >= 0.4:
        return "medio"
    return "baixo"


def predict_churn(customers: list[dict], artifact: dict | None = None) -> list[dict]:
    """Recebe uma lista de clientes (dicionários) e devolve a propensão de churn."""
    artifact = artifact or load_model()
    df = clean_data(pd.DataFrame(customers))[artifact["features"]]
    probabilities = artifact["pipeline"].predict_proba(df)[:, 1]
    threshold = artifact["threshold"]

    return [
        {
            "churn_probability": round(float(p), 4),
            "churn_prediction": bool(p >= threshold),
            "risk_level": risk_level(float(p)),
        }
        for p in probabilities
    ]
