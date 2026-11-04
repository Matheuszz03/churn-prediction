import numpy as np
import pandas as pd

from src.config import FEATURES
from src.evaluation import business_cost, cost_without_model
from src.preprocessing import build_preprocessor, clean_data


def test_clean_data_converts_blank_total_charges_to_zero(sample_customer):
    raw = pd.DataFrame([{**sample_customer, "TotalCharges": " ", "tenure": 0}])
    cleaned = clean_data(raw)
    assert cleaned["TotalCharges"].iloc[0] == 0.0
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])


def test_clean_data_drops_id_and_encodes_target(sample_customer):
    raw = pd.DataFrame(
        [
            {**sample_customer, "customerID": "0001-A", "Churn": "Yes"},
            {**sample_customer, "customerID": "0002-B", "Churn": "No"},
        ]
    )
    cleaned = clean_data(raw)
    assert "customerID" not in cleaned.columns
    assert cleaned["Churn"].tolist() == [1, 0]


def test_preprocessor_outputs_numeric_matrix_without_nans(sample_customer):
    other = {**sample_customer, "gender": "Male", "Contract": "Two year", "tenure": 60}
    df = clean_data(pd.DataFrame([sample_customer, other]))[FEATURES]
    matrix = build_preprocessor().fit_transform(df)
    assert matrix.shape[0] == 2
    assert not np.isnan(np.asarray(matrix, dtype=float)).any()


def test_business_cost_perfect_model_is_cheaper_than_doing_nothing():
    y_true = np.array([1, 0, 1, 0])
    charges = np.array([100.0, 50.0, 80.0, 70.0])
    perfect = business_cost(y_true, y_true.astype(float), charges, threshold=0.5)
    assert perfect < cost_without_model(y_true, charges)
