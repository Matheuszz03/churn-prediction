"""API de inferência de churn.

Uso:
    uvicorn src.api:app --reload
Documentação interativa em http://127.0.0.1:8000/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from src.predict import load_model, predict_churn
from src.schemas import Customer, HealthResponse, PredictionResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("churn-api")

state: dict = {"artifact": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Carrega o modelo uma vez, na subida da API."""
    try:
        state["artifact"] = load_model()
        logger.info("Modelo carregado: %s", state["artifact"]["model_name"])
    except FileNotFoundError as exc:
        logger.error(str(exc))
    yield


app = FastAPI(
    title="Churn Prediction API",
    description="Prevê a propensão de cancelamento de clientes de telecom.",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redireciona a raiz para a documentação interativa."""
    return RedirectResponse(url="/docs")

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Verifica se a API está no ar e se o modelo foi carregado."""
    artifact = state["artifact"]
    return HealthResponse(
        status="ok",
        model_loaded=artifact is not None,
        model_name=artifact["model_name"] if artifact else None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: Customer) -> PredictionResponse:
    """Recebe os dados de um cliente e retorna a propensão de churn."""
    artifact = state["artifact"]
    if artifact is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")

    result = predict_churn([customer.model_dump()], artifact)[0]
    logger.info("Predição: prob=%.4f risco=%s", result["churn_probability"], result["risk_level"])
    return PredictionResponse(
        **result,
        model_name=artifact["model_name"],
        threshold=artifact["threshold"],
    )
