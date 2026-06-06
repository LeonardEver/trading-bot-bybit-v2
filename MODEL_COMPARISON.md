OBJETIVO

Validar a hipótese de que o modelo possui mais capacidade
de prever "oportunidades operáveis" do que direção.

EXPERIMENTO 001

Criar um novo dataset derivado do Triple Barrier.

Classe 0:
Neutro

Classe 1:
TP ou SL

Treinar LightGBM usando exatamente as mesmas features,
mesmos hiperparâmetros e mesmo TimeSeriesSplit.

Gerar:

- ROC AUC
- PR AUC
- Brier Score
- Calibration Curve
- SHAP Importance
- Fold-by-fold metrics

Comparar com o modelo Triple Barrier atual.

Critério de aprovação:

AUC >= 0.70

Entregar relatório completo
tradeability_model_report.md