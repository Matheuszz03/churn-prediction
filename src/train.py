"""Treina, compara e salva o modelo campeão de churn.

Uso:
    python -m src.train
"""

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)

from src.config import (
    CV_FOLDS,
    EXPERIMENTS_PATH,
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.evaluation import (
    business_cost,
    cost_without_model,
    find_best_threshold,
    technical_metrics,
)
from src.models import build_pipeline, get_models
from src.preprocessing import clean_data, load_raw_data, split_features_target

SCORING = ["roc_auc", "f1", "recall", "precision"]


def set_seeds(seed: int = RANDOM_STATE) -> None:
    """Fixa as seeds globais para reprodutibilidade."""
    np.random.seed(seed)


def load_and_split():
    """Carrega, limpa e separa treino/teste de forma estratificada."""
    df = clean_data(load_raw_data())
    X, y = split_features_target(df)
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def compare_models(X_train, y_train) -> tuple[pd.DataFrame, dict]:
    """Avalia cada modelo com validação cruzada estratificada no treino.

    Retorna a tabela de resultados e as probabilidades out-of-fold de cada
    modelo (usadas para escolher o limiar de decisão sem tocar no teste).
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows, oof_probas = [], {}
    baseline_cost = cost_without_model(y_train, X_train["MonthlyCharges"])

    for name, model in get_models().items():
        pipeline = build_pipeline(model)
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=SCORING)
        oof = cross_val_predict(
            pipeline, X_train, y_train, cv=cv, method="predict_proba"
        )[:, 1]
        oof_probas[name] = oof

        best_t, _ = find_best_threshold(y_train, oof, X_train["MonthlyCharges"])
        cost = business_cost(y_train, oof, X_train["MonthlyCharges"], best_t)

        row = {"model": name}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"cv_{metric}_mean"] = values.mean()
            row[f"cv_{metric}_std"] = values.std()
        row["best_threshold"] = best_t
        row["oof_business_saving_pct"] = 1 - cost / baseline_cost
        rows.append(row)
        print(
            f"{name:<20} AUC={row['cv_roc_auc_mean']:.4f} ± {row['cv_roc_auc_std']:.4f}"
            f" | F1={row['cv_f1_mean']:.4f} | economia={row['oof_business_saving_pct']:.1%}"
        )

    results = pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False)
    return results.reset_index(drop=True), oof_probas


def main() -> None:
    set_seeds()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_and_split()
    print(f"Treino: {len(X_train)} | Teste: {len(X_test)} | Churn treino: {y_train.mean():.1%}\n")

    print(f"Validação cruzada ({CV_FOLDS} folds):")
    results, _ = compare_models(X_train, y_train)

    # Campeão = maior AUC-ROC média na validação cruzada
    champion = results.iloc[0]
    champion_name = champion["model"]
    threshold = float(champion["best_threshold"])
    print(f"\nModelo campeão: {champion_name} (limiar de decisão = {threshold})")

    # Avaliação final no conjunto de teste (dados nunca vistos)
    test_rows = []
    fitted = {}
    for name, model in get_models().items():
        pipeline = build_pipeline(model).fit(X_train, y_train)
        fitted[name] = pipeline
        proba = pipeline.predict_proba(X_test)[:, 1]
        t = float(results.loc[results["model"] == name, "best_threshold"].iloc[0])
        metrics = technical_metrics(y_test, proba, t)
        cost = business_cost(y_test, proba, X_test["MonthlyCharges"], t)
        metrics["business_saving_pct"] = 1 - cost / cost_without_model(
            y_test, X_test["MonthlyCharges"]
        )
        test_rows.append({"model": name, **{f"test_{k}": v for k, v in metrics.items()}})

    test_results = pd.DataFrame(test_rows)
    experiments = results.merge(test_results, on="model")
    experiments.insert(0, "run_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    experiments.to_csv(
        EXPERIMENTS_PATH, mode="a", header=not EXPERIMENTS_PATH.exists(), index=False
    )

    champion_test = test_results.set_index("model").loc[champion_name].to_dict()
    artifact = {
        "pipeline": fitted[champion_name],
        "model_name": champion_name,
        "threshold": threshold,
        "features": FEATURES,
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(artifact, MODEL_PATH)

    summary = {
        "champion": champion_name,
        "threshold": threshold,
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS,
        "test_metrics": {k: round(float(v), 4) for k, v in champion_test.items()},
        "comparison": experiments.drop(columns=["run_at"]).round(4).to_dict(orient="records"),
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\nResultados no teste (limiar otimizado por modelo):")
    print(test_results.round(4).to_string(index=False))
    print(f"\nModelo salvo em {MODEL_PATH.relative_to(MODELS_DIR.parent)}")


if __name__ == "__main__":
    main()
