# Model Card — Classificador de Churn

## Detalhes do modelo

| Item | Valor |
|---|---|
| Tipo | Random Forest (Scikit-Learn `RandomForestClassifier`) dentro de um `Pipeline` com pré-processamento |
| Hiperparâmetros | 300 árvores, `max_depth=8`, `min_samples_leaf=5`, `class_weight="balanced"` |
| Limiar de decisão | **0,46** (minimiza o custo de negócio nas previsões out-of-fold do treino) |
| Seed | 42 |
| Artefato | `models/churn_model.joblib` (pipeline + limiar + lista de features) |
| Versão do Scikit-Learn | 1.9.1 (o `.joblib` deve ser carregado com a mesma versão) |

## Uso pretendido

- **Para:** priorizar clientes ativos para ações de retenção da operadora.
- **Usuários:** time de Retenção / CRM, via API.
- **Não usar para:** negar serviço, mudar preço individual ou qualquer decisão automática que
  prejudique o cliente. O modelo indica **risco**, não causa, e a decisão final deve ser humana.

## Dados

- Telco Customer Churn (IBM), 7.043 clientes, 26,5% de churn.
- Divisão estratificada: 80% treino (5.634) e 20% teste (1.409).
- Limpeza: 11 valores em branco de `TotalCharges` (clientes com `tenure = 0`) preenchidos com 0.

## Desempenho

### Comparação (validação cruzada de 5 folds no treino)

| Modelo | AUC-ROC | F1 (limiar 0,5) |
|---|---|---|
| **Random Forest** | **0,848 ± 0,010** | 0,632 |
| Regressão Logística (baseline) | 0,846 ± 0,012 | 0,629 |
| MLP | 0,842 ± 0,014 | 0,577 |

### Modelo campeão no teste (limiar 0,46)

| Métrica | Valor |
|---|---|
| AUC-ROC | 0,843 |
| Recall | 0,818 (306 de 374 clientes que cancelaram foram identificados) |
| Precision | 0,500 (metade dos abordados realmente cancelaria) |
| F1 | 0,621 |
| Clientes sinalizados | 43% da base |
| **Economia vs não agir** | **22,7%** do custo de churn |

## Variáveis mais importantes

Contrato mensal, tempo de casa (`tenure`), valor total pago, contrato de 2 anos, ausência de
segurança online e de suporte técnico, valor mensal, internet por fibra e pagamento por boleto
eletrônico. São consistentes com a EDA e com a intuição de negócio.

## Limitações

1. **Ganho pequeno sobre o baseline.** A Random Forest supera a Regressão Logística por apenas
   0,002 de AUC, dentro da variação entre folds. No teste, a Regressão Logística teve economia de
   negócio um pouco maior (23,7%). Se interpretabilidade for prioridade, ela é uma alternativa válida.
2. **Precision de 50%.** Metade das ofertas vai para clientes que não cancelariam. Isso é aceitável
   pelas premissas de custo atuais, mas fica caro se a oferta de retenção for cara.
3. **Premissas de custo assumidas.** 12 meses de receita perdida, R$ 100 por oferta e 50% de sucesso
   na retenção são estimativas. Precisam ser validadas com dados reais de campanhas, e o limiar deve
   ser recalculado quando mudarem.
4. **Dados estáticos e antigos.** Um único recorte, sem dimensão temporal nem informações de uso
   (consumo, chamados ao SAC, reclamações), que costumam ser fortes preditores de churn.
5. **Correlação, não causa.** O modelo não diz *por que* o cliente vai sair nem qual ação o retém.
6. **Probabilidades não calibradas.** O `class_weight="balanced"` desloca as probabilidades para
   cima; o valor de `churn_probability` deve ser lido como um **escore de risco** para ordenar
   clientes, não como probabilidade real.

## Vieses e considerações éticas

- **Gênero:** desempenho equivalente entre homens e mulheres (AUC 0,841 vs 0,846). Sem sinal de viés.
- **Idosos (`SeniorCitizen=1`):** o modelo é **menos preciso** para esse grupo (AUC 0,78 vs 0,84) e
  sinaliza 68% deles. Parte disso reflete o churn real maior (44%), mas o grupo é pequeno (16% da base)
  e merece monitoramento específico.
- Variáveis demográficas (`gender`, `SeniorCitizen`, `Partner`, `Dependents`) foram mantidas porque
  o uso é de retenção (benéfico ao cliente). Para qualquer uso que possa prejudicar o cliente, elas
  devem ser removidas e o modelo reavaliado.

## Manutenção

Retreinar quando a AUC mensal cair abaixo de 0,80 ou quando houver mudança relevante de portfólio
(novos planos, preços). Comando: `python -m src.train`.
