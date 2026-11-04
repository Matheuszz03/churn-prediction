# ML Canvas — Previsão de Churn

| Bloco | Descrição |
|---|---|
| **Proposta de valor** | Identificar, com antecedência, os clientes com maior risco de cancelamento para que o time de retenção atue antes da saída, reduzindo a perda de receita. |
| **Stakeholders** | Diretoria (patrocinador, acompanha receita e churn); time de Retenção / CRM (usa a lista de clientes em risco); Marketing (desenha as ofertas); Atendimento (executa o contato); Time de Dados (mantém o modelo). |
| **Tarefa de ML** | Classificação binária supervisionada: o cliente vai cancelar (1) ou não (0)? A saída é uma probabilidade, usada para priorizar clientes. |
| **Decisões** | Clientes com probabilidade acima do limiar recebem uma ação de retenção (oferta, contato proativo, migração para contrato anual). A faixa de risco (baixo/médio/alto) ajuda a priorizar quem contatar primeiro. |
| **Fontes de dados** | Base cadastral e de faturamento: perfil demográfico, serviços contratados, tipo de contrato, forma de pagamento, valores cobrados e tempo de casa. Dataset Telco Customer Churn (IBM), 7.043 clientes. |
| **Coleta de dados** | Histórico mensal de clientes, com o rótulo de churn observado no mês seguinte. Em produção, a base seria atualizada mensalmente a partir do CRM e do faturamento. |
| **Features** | 19 variáveis: 3 numéricas (tenure, MonthlyCharges, TotalCharges), 1 binária (SeniorCitizen) e 15 categóricas (contrato, serviços, pagamento etc.). |
| **Construção do modelo** | Pipeline Scikit-Learn (pré-processamento + modelo). Comparação de Regressão Logística (baseline), Random Forest e MLP com validação cruzada estratificada de 5 folds. Seed fixa = 42. |
| **Predições** | Sob demanda via API REST (`POST /predict`), um cliente por chamada. Também pode rodar em lote mensal sobre a base ativa. |
| **Avaliação offline** | Técnica: **AUC-ROC** (critério de escolha), recall, precision e F1. Negócio: **economia (%) em relação a não agir**, considerando perda de 12 mensalidades por churn não detectado e R$ 100 por oferta de retenção (50% de sucesso). |
| **Monitoramento** | Acompanhar mensalmente: AUC e recall com os rótulos reais; distribuição das features (drift), principalmente MonthlyCharges e tipo de contrato; taxa de clientes abordados; resultado real das campanhas de retenção para recalibrar as premissas de custo. |
| **Métricas de negócio** | Taxa de churn mensal; receita retida; custo da campanha de retenção; ROI da campanha. |
