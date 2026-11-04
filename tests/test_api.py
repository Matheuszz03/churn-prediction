import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture(scope="module")
def client():
    # O "with" dispara o lifespan, que carrega o modelo
    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_valid_probability(client, sample_customer):
    response = client.post("/predict", json=sample_customer)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_level"] in {"baixo", "medio", "alto"}
    assert body["churn_prediction"] == (body["churn_probability"] >= body["threshold"])


def test_predict_high_risk_profile_scores_higher_than_loyal_profile(client, sample_customer):
    loyal = {
        **sample_customer,
        "tenure": 70,
        "Contract": "Two year",
        "PaymentMethod": "Credit card (automatic)",
        "TechSupport": "Yes",
        "OnlineSecurity": "Yes",
        "TotalCharges": 6700.0,
    }
    risky = client.post("/predict", json=sample_customer).json()["churn_probability"]
    safe = client.post("/predict", json=loyal).json()["churn_probability"]
    assert risky > safe


def test_predict_rejects_invalid_payload(client, sample_customer):
    invalid = {**sample_customer, "Contract": "Lifetime"}
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422
