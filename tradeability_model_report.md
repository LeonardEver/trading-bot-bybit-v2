# Tradeability Model — Relatorio do Experimento 001

**Data:** 2026-06-06 14:09:52
**Hipotese:** O modelo tem mais capacidade de prever "oportunidades operaveis"
(tradeability) do que direcao (TP vs SL).

**Target:**
- Classe 0 = Neutro (nenhuma barreira atingida em 12 candles)
- Classe 1 = Tradeable (TP ou SL atingido)

**Parametros:** LightGBM identico ao Triple Barrier v2 | TimeSeriesSplit 5 folds

---

## 1. Resultado Principal

| Metrica | Tradeability | TB Direction | Delta |
|---------|-------------|--------------|-------|
| **ROC AUC** | **0.7963** | 0.5369 | +0.2594 |
| **PR AUC** | **0.8993** | 0.3011 | +0.5982 |
| **Accuracy** | 0.7531 | 0.6940 | — |
| **LogLoss** | 0.4980 | 0.6346 | — |
| **Brier Score** | 0.1652 | 0.2142 | -0.0491 |

### Criterio de Aprovacao: AUC >= 0.70

| Criterio | Threshold | Observado | Status |
|----------|-----------|-----------|--------|
| ROC AUC | >= 0.70 | 0.7963 | ✅ APROVADO |

---

## 2. Metricas por Fold

| Fold | AUC | PR AUC | Accuracy | LogLoss | Brier |
|------|-----|--------|----------|---------|-------|
| 0 | 0.7326 | 0.7828 | 0.6669 | 0.6233 | 0.2132 |
| 1 | 0.7676 | 0.9221 | 0.7701 | 0.4590 | 0.1522 |
| 2 | 0.8464 | 0.9175 | 0.7651 | 0.4593 | 0.1531 |
| 3 | 0.7918 | 0.9447 | 0.8494 | 0.3883 | 0.1169 |
| 4 | 0.7667 | 0.8593 | 0.7140 | 0.5601 | 0.1905 |

- **AUC Media:** 0.7810 +/- 0.0378
- **PR AUC Media:** 0.8853 +/- 0.0585

---

## 3. SHAP Feature Importance (Tradeability)

| Rank | Feature | Mean(|SHAP|) |
|------|---------|---------------|
| 1 | `atr_pct` | 1.929978 |
| 2 | `fracdiff_close` | 0.373827 |
| 3 | `ema_200_return` | 0.332152 |
| 4 | `high_low_log_range` | 0.278805 |
| 5 | `hour` | 0.172188 |
| 6 | `anchored_vwap_distance` | 0.121042 |
| 7 | `volume_ma_ratio` | 0.102728 |
| 8 | `macd_signal_return` | 0.091356 |
| 9 | `log_return` | 0.082589 |
| 10 | `session_volume_share` | 0.079321 |
| 11 | `vwap_deviation_zscore` | 0.076197 |
| 12 | `log_return_3` | 0.074234 |
| 13 | `macd_hist_return` | 0.072191 |
| 14 | `ema_20_return` | 0.071360 |
| 15 | `ema_50_return` | 0.065669 |
| 16 | `volume_log_change` | 0.062467 |
| 17 | `log_return_5` | 0.059230 |
| 18 | `macd_return` | 0.058400 |
| 19 | `log_return_15` | 0.054461 |
| 20 | `fracdiff_close_5` | 0.042462 |

---

## 4. Comparacao Visual

| Grafico | Arquivo |
|---------|---------|
| ROC Curves | `ml/triple_barrier_report/roc_tradeability.png` |
| PR Curves | `ml/triple_barrier_report/pr_tradeability.png` |
| Calibration | `ml/triple_barrier_report/calibration_tradeability.png` |
| SHAP | `ml/triple_barrier_report/shap_tradeability.png` |
| Fold Metrics | `ml/triple_barrier_report/fold_metrics_tradeability.png` |

---

## 5. Interpretacao

### 5.1 A hipotese se confirma?

**SIM, fortemente.** O modelo de tradeability apresenta AUC significativamente superior (0.7963 vs 0.5369, delta = +0.2594). Isso confirma que o modelo tem mais capacidade de identificar volatilidade/oportunidade do que direcao.

### 5.2 Implicacoes Praticas

- Um modelo de tradeability com AUC=0.7963 pode ser usado para:
  - **Filtro de entrada:** evitar operar em candles com baixa probabilidade de atingir barreiras
  - **Gestao de risco:** reduzir exposicao em periodos de baixa volatilidade prevista
  - **Sizeamento:** aumentar posicao quando tradeability e alta

### 5.3 Limitacoes

- O target e definido pelos mesmos parametros do Triple Barrier (TP +0.40%, SL -0.20%, 12 candles)
- A classe 1 (tradeable) inclui tanto TP quanto SL — nao distingue oportunidades boas de ruins
- Performance em producao depende da estabilidade do regime de volatilidade

---

## 6. Conclusao

**APROVADO** — AUC de 0.7963 atinge o criterio minimo de 0.70. O modelo de tradeability demonstra capacidade preditiva util e pode ser integrado ao pipeline como filtro complementar ao modelo de direcao. Recomenda-se paper trading com o modelo combinado (tradeability gate + direction model).

---

*Relatorio gerado automaticamente — 2026-06-06 14:09:52*
*Modelo: LightGBM | Features: identicas ao v1 | Target: Tradeability (Neutro vs Tradeable)*
