"""Definição dos modelos comparados no projeto."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.preprocessing import build_preprocessor


def get_models(random_state: int = RANDOM_STATE) -> dict:
    """Retorna os três modelos candidatos, sem pré-processamento.

    - Regressão Logística: baseline linear e interpretável.
    - Random Forest: ensemble de árvores, captura não linearidades e interações.
    - MLP: rede neural simples (duas camadas ocultas) com early stopping.

    class_weight="balanced" compensa o desbalanceamento (~26% de churn) nos
    modelos que suportam o parâmetro. Para o MLP, o desbalanceamento é tratado
    no ajuste do limiar de decisão.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=random_state,
        ),
    }


def build_pipeline(model) -> Pipeline:
    """Encadeia pré-processamento + modelo em um único objeto treinável.

    Usar Pipeline evita vazamento de dados na validação cruzada, porque o
    scaler e o encoder são ajustados apenas nos dados de treino de cada fold.
    """
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", model)])
