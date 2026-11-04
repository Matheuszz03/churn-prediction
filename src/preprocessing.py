"""Carregamento, limpeza e pré-processamento dos dados."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    BINARY_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FEATURES,
    ID_COLUMN,
    NUMERIC_FEATURES,
    TARGET,
)


def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Lê o CSV bruto do dataset Telco Customer Churn."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica as correções de qualidade identificadas na EDA.

    - TotalCharges vem como texto e tem 11 valores em branco, todos de clientes
      com tenure = 0 (ainda não pagaram nenhuma fatura). Convertemos para número
      e preenchemos com 0, que é o valor correto para esses clientes.
    - Remove o identificador do cliente, que não tem poder preditivo.
    - Converte o alvo Churn de "Yes"/"No" para 1/0.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    if TARGET in df.columns and df[TARGET].dtype != "int64":
        df[TARGET] = (df[TARGET] == "Yes").astype(int)

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa a matriz de features (X) e o alvo (y)."""
    return df[FEATURES], df[TARGET]


def build_preprocessor() -> ColumnTransformer:
    """Cria o transformador de colunas usado por todos os modelos.

    Numéricas: imputação pela mediana + padronização (importante para
    Regressão Logística e MLP, que são sensíveis à escala).
    Categóricas: one-hot encoding, ignorando categorias novas na inferência.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
