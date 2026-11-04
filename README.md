# Previsão de Churn — Telecom

Pipeline de Machine Learning para identificar clientes de uma operadora de telecomunicações com
risco de cancelamento, da análise exploratória até um modelo servido por API REST.

**Tech Challenge — Fase 1 (Pós-Tech Machine Learning Engineering)**

## Resultados

| Modelo | AUC-ROC (CV 5 folds) | AUC-ROC (teste) | Recall (teste) | Economia vs não agir |
|---|---|---|---|---|
| **Random Forest (campeão)** | **0,848 ± 0,010** | 0,843 | 0,818 | 22,7% |
| Regressão Logística (baseline) | 0,846 ± 0,012 | 0,842 | 0,778 | 23,7% |
| MLP (rede neural) | 0,842 ± 0,014 | 0,841 | 0,789 | 23,1% |

O modelo campeão identifica **82% dos clientes que vão cancelar** e reduz em **~23%** o custo
estimado de churn em relação a não fazer nada. Os três modelos ficaram muito próximos; a discussão
completa está em [`notebooks/02_modeling.ipynb`](notebooks/02_modeling.ipynb) e no
[Model Card](docs/model_card.md).

## Estrutura

```
├── data/
│   └── raw/telco_churn.csv     # Dataset Telco Customer Churn (IBM)
├── docs/
│   ├── ml_canvas.md            # Formulação do problema e stakeholders
│   ├── model_card.md           # Desempenho, limitações e vieses
│   └── figures/                # Gráficos gerados pelos notebooks
├── models/
│   ├── churn_model.joblib      # Pipeline do modelo campeão + limiar
│   ├── metrics.json            # Métricas do último treino
│   └── experiments.csv         # Registro de experimentos
├── notebooks/
│   ├── 01_eda_baseline.ipynb   # EDA, definição de métricas e baseline
│   └── 02_modeling.ipynb       # Comparação de modelos e escolha do campeão
├── src/
│   ├── config.py               # Caminhos, seed e premissas de negócio
│   ├── preprocessing.py        # Limpeza e pré-processamento
│   ├── models.py               # Definição dos modelos
│   ├── evaluation.py           # Métricas técnicas e de negócio
│   ├── train.py                # Treino, comparação e salvamento
│   ├── predict.py              # Carregamento do modelo e predição
│   ├── schemas.py              # Contratos da API (Pydantic)
│   └── api.py                  # API FastAPI
├── tests/                      # Testes com pytest
├── pyproject.toml
└── requirements.txt
```

Os **notebooks** são usados para experimentação e análise. Todo o código reaproveitável fica em
**`src/`** e é importado pelos notebooks, pela API e pelos testes.

## Como rodar

Requisitos: Python 3.11 a 3.13.

### 1. Instalar as dependências

```bash
git clone <url-do-repositorio>
cd churn-prediction

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Treinar o modelo

```bash
python -m src.train
```

Treina os três modelos com validação cruzada, escolhe o campeão pela AUC-ROC, avalia no teste e
salva o modelo em `models/churn_model.joblib`. Leva menos de 1 minuto. O modelo já treinado está
versionado, então este passo é opcional.

### 3. Subir a API

```bash
uvicorn src.api:app --reload
```

Documentação interativa (Swagger): http://127.0.0.1:8000/docs

### 4. Testar os endpoints

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status": "ok", "model_loaded": true, "model_name": "random_forest"}
```

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.7, "TotalCharges": 191.4
  }'
```

```json
{
  "churn_probability": 0.8825,
  "churn_prediction": true,
  "risk_level": "alto",
  "model_name": "random_forest",
  "threshold": 0.46
}
```

Campos inválidos (ex.: `"Contract": "Lifetime"`) retornam **422** com a descrição do erro.

### 5. Rodar os testes

```bash
pytest
```

### 6. Reproduzir os notebooks

```bash
jupyter notebook notebooks/
```

## Decisões técnicas

- **Métricas:** AUC-ROC como critério principal (robusta ao desbalanceamento de 26,5% de churn),
  com recall, precision e F1 de apoio. Métrica de negócio: economia em R$ em relação a não agir,
  com premissas definidas em `src/config.py`.
- **Desbalanceamento:** `class_weight="balanced"` na Regressão Logística e na Random Forest, e
  limiar de decisão ajustado para minimizar o custo de negócio em todos os modelos.
- **Sem vazamento de dados:** pré-processamento dentro do `Pipeline`, ajustado só no treino de
  cada fold. O limiar é escolhido com previsões out-of-fold; o teste é usado apenas na avaliação final.
- **Reprodutibilidade:** seed 42 fixa em split, validação cruzada e modelos.
- **API:** validação de entrada com Pydantic (tipos e valores permitidos), modelo carregado uma
  única vez na inicialização e logging das predições.

## Limitações

Resumo (detalhes no [Model Card](docs/model_card.md)): ganho pequeno sobre o baseline, precision
de 50%, premissas de custo estimadas, dados estáticos sem informação de uso, e desempenho inferior
para clientes idosos.
