Objetivo: Implementar e validar um novo Target baseado em Triple Barrier Method sem alterar nenhuma feature ou hiperparâmetro do modelo atual.

Regras obrigatórias:

1. NÃO adicionar novas features.
2. NÃO alterar hiperparâmetros do LightGBM.
3. NÃO alterar TimeSeriesSplit.
4. NÃO alterar Walk Forward Validation.
5. Preservar integralmente o pipeline atual para comparação A/B.

Implementação:

Barreira Superior:
+0.40%

Barreira Inferior:
-0.20%

Barreira Temporal:
12 candles (60 minutos)

Label:

* target = 1 se Take Profit for atingido antes do Stop Loss
* target = 0 se Stop Loss for atingido antes do Take Profit
* target = NaN caso nenhuma barreira seja atingida antes da barreira temporal

Criar:

* ml_data_pipeline_v2.py
* dataset_triple_barrier.csv

Executar treinamento usando exatamente o mesmo modelo LightGBM atual.

Registrar:

* ROC AUC
* Accuracy
* LogLoss
* Brier Score
* Calibration Curve

Executar SHAP Values após o treinamento.

Produzir relatório final contendo:

1. Comparação entre Target Atual e Triple Barrier.
2. Evolução do ROC AUC.
3. Evolução do LogLoss.
4. Ranking de Features.
5. Conclusão estatística sobre a validade do Triple Barrier.
6. Recomendação objetiva:

   * Aprovar migração
   * Rejeitar migração
   * Necessário mais testes
