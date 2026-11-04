# Dados

`raw/telco_churn.csv` — **Telco Customer Churn (IBM)**, também disponível no Kaggle.

- 7.043 clientes, 21 colunas (19 features + ID + alvo `Churn`)
- Fonte: https://github.com/IBM/telco-customer-churn-on-icp4d (pasta `data/`)
- Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

O arquivo é pequeno (~1 MB) e foi versionado no repositório para garantir reprodutibilidade.
Para baixá-lo novamente:

```bash
curl -L -o data/raw/telco_churn.csv \
  https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```

## Dicionário

| Coluna | Descrição |
|---|---|
| gender, SeniorCitizen, Partner, Dependents | Perfil demográfico |
| tenure | Meses como cliente |
| PhoneService, MultipleLines | Telefonia |
| InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies | Internet e serviços adicionais |
| Contract, PaperlessBilling, PaymentMethod | Contrato e cobrança |
| MonthlyCharges, TotalCharges | Valor mensal e total pago |
| Churn | Alvo: cancelou no último mês (Yes/No) |
