"""Configurações centrais do projeto: caminhos, seed e premissas de negócio."""

from pathlib import Path

# Caminhos
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "telco_churn.csv"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
EXPERIMENTS_PATH = MODELS_DIR / "experiments.csv"

# Reprodutibilidade
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Colunas
ID_COLUMN = "customerID"
TARGET = "Churn"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_NUMERIC_FEATURES = ["SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + BINARY_NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Premissas da métrica de negócio (custo de erros de classificação).
# Falso negativo: cliente que sai sem ser abordado -> perde-se ~12 meses de receita.
# Falso positivo: cliente fiel que recebe oferta de retenção sem necessidade.
MONTHS_OF_REVENUE_LOST = 12
RETENTION_OFFER_COST = 100.0  # R$ por cliente abordado
RETENTION_SUCCESS_RATE = 0.5  # fração dos clientes em risco abordados que ficam
