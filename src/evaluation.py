"""Métricas técnicas e métrica de negócio."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    MONTHS_OF_REVENUE_LOST,
    RETENTION_OFFER_COST,
    RETENTION_SUCCESS_RATE,
)


def technical_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Calcula as métricas técnicas para um limiar de decisão."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def business_cost(y_true, y_proba, monthly_charges, threshold: float = 0.5) -> float:
    """Custo total (R$) de usar o modelo para direcionar ofertas de retenção.

    - Falso negativo: o cliente sai sem ser abordado -> perde-se a receita
      de MONTHS_OF_REVENUE_LOST meses daquele cliente.
    - Verdadeiro positivo: o cliente recebe a oferta (custo fixo) e só uma
      parte (RETENTION_SUCCESS_RATE) é retida; o restante ainda é perdido.
    - Falso positivo: paga-se a oferta para um cliente que não sairia.
    """
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    revenue = np.asarray(monthly_charges) * MONTHS_OF_REVENUE_LOST

    fn = (y_true == 1) & (y_pred == 0)
    tp = (y_true == 1) & (y_pred == 1)
    fp = (y_true == 0) & (y_pred == 1)

    cost_fn = revenue[fn].sum()
    cost_tp = revenue[tp].sum() * (1 - RETENTION_SUCCESS_RATE) + tp.sum() * RETENTION_OFFER_COST
    cost_fp = fp.sum() * RETENTION_OFFER_COST
    return float(cost_fn + cost_tp + cost_fp)


def cost_without_model(y_true, monthly_charges) -> float:
    """Custo de não fazer nada: toda a receita dos clientes que saem é perdida."""
    revenue = np.asarray(monthly_charges) * MONTHS_OF_REVENUE_LOST
    return float(revenue[np.asarray(y_true) == 1].sum())


def find_best_threshold(y_true, y_proba, monthly_charges) -> tuple[float, pd.DataFrame]:
    """Procura o limiar que minimiza o custo de negócio.

    Deve ser aplicado sobre previsões out-of-fold do treino, nunca no teste.
    """
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    costs = [business_cost(y_true, y_proba, monthly_charges, t) for t in thresholds]
    curve = pd.DataFrame({"threshold": thresholds, "cost": costs})
    best = float(curve.loc[curve["cost"].idxmin(), "threshold"])
    return best, curve
