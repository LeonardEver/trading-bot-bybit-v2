# Triple Barrier Method — Relatório de Avaliação

**Data:** 2026-06-06 13:41:43
**Modelo:** LightGBM (parâmetros idênticos ao target atual)
**Validação:** TimeSeriesSplit (5 folds)

---

## 1. Resumo Comparativo

| Métrica          | Target Atual (v1) | Triple Barrier (v2) | Diferença (v2 − v1) | Vencedor       |
|------------------|--------------------|----------------------|----------------------|----------------|
| **ROC AUC**      | 0.5104    | 0.5396      | +0.0292     | Triple Barrier   |
| **Accuracy**     | 0.5073 | 0.6949 | +0.1877     | —              |
| **LogLoss**      | 0.7096 | 0.6330 | -0.0766      | Triple Barrier    |
| **Brier Score**  | 0.2574 | 0.2136 | -0.0438   | Triple Barrier |

---

## 2. Evolução por Fold (TimeSeriesSplit)

### 2.1 ROC AUC

| Fold | Target Atual | Triple Barrier | Delta  |
|------|-------------|----------------|--------|
| 0    | 0.5026     | 0.5306       | +0.0280 |
| 1    | 0.5077     | 0.5130       | +0.0053 |
| 2    | 0.5158     | 0.5604       | +0.0447 |
| 3    | 0.5132     | 0.5355       | +0.0223 |
| 4    | 0.5172     | 0.5526       | +0.0353 |

### 2.2 LogLoss

| Fold | Target Atual | Triple Barrier | Delta  |
|------|-------------|----------------|--------|
| 0    | 0.7269     | 0.6878       | -0.0392 |
| 1    | 0.7132     | 0.6544       | -0.0588 |
| 2    | 0.7049     | 0.6146       | -0.0903 |
| 3    | 0.7039     | 0.6319       | -0.0720 |
| 4    | 0.6989     | 0.5763       | -0.1226 |

---

## 3. Ranking de Features (SHAP — Triple Barrier)

| Rank | Feature | Mean(\|SHAP\|) | Rank v1 |
|------|---------|---------------|---------|
| 8 | `fracdiff_close` | 0.406409 | 1 |
| 2 | `atr_pct` | 0.246585 | 7 |
| 5 | `ema_200_return` | 0.141549 | 3 |
| 37 | `vwap_deviation_zscore` | 0.131498 | 14 |
| 13 | `hour` | 0.110355 | 21 |
| 20 | `log_return_15` | 0.078312 | 12 |
| 23 | `macd_hist_return` | 0.072843 | 13 |
| 25 | `macd_signal_return` | 0.070282 | 9 |
| 1 | `anchored_vwap_distance` | 0.068117 | 5 |
| 6 | `ema_20_return` | 0.064889 | 2 |
| 3 | `close_open_log_return` | 0.056209 | 18 |
| 12 | `high_low_log_range` | 0.056152 | 10 |
| 7 | `ema_50_return` | 0.054649 | 8 |
| 22 | `log_return_5` | 0.054059 | 15 |
| 36 | `volume_ma_ratio` | 0.052154 | 19 |
| 35 | `volume_log_change` | 0.050838 | 11 |
| 21 | `log_return_3` | 0.050361 | 16 |
| 33 | `session_volume_share` | 0.048287 | 17 |
| 9 | `fracdiff_close_5` | 0.046890 | 4 |
| 24 | `macd_return` | 0.044878 | 6 |

---

## 4. Overlap de Features Importantes

Top-5 features **Target Atual**: `fracdiff_close`, `ema_20_return`, `ema_200_return`, `fracdiff_close_5`, `anchored_vwap_distance`
Top-5 features **Triple Barrier**: `fracdiff_close`, `atr_pct`, `ema_200_return`, `vwap_deviation_zscore`, `hour`
Overlap no Top-5: **2/5**

---

## 5. Conclusão Estatística

### 5.1 Critérios de Avaliação

| Critério                          | Threshold        | Valor Observado | Status |
|-----------------------------------|------------------|-----------------|--------|
| AUC Triple Barrier ≥ AUC Atual    | Δ ≥ −0.01        | +0.0292 | ✅ |
| Brier Score não degradado         | Δ ≤ +0.02        | -0.0438 | ✅ |
| Overlap SHAP Top-5 ≥ 2            | ≥ 2              | 2/5 | ✅ |

### 5.2 Interpretação

- **ROC AUC**: O Triple Barrier apresenta AUC superior ao target atual.
- **Calibração**: O Brier Score é melhor (menor) no Triple Barrier.
- **Features**: O ranking de features mantém consistência com o target atual (overlap de 2/5 no Top-5).

---

## 6. Recomendação

✅ **Aprovar migração** — Triple Barrier é superior ou equivalente em todas as métricas relevantes.

### Fundamentação:

O Triple Barrier Method demonstrou desempenho superior ou equivalente ao target atual em todas as métricas de avaliação (ROC AUC, LogLoss, Brier Score). As features mais importantes mantêm consistência, indicando que o modelo está capturando sinais financeiramente interpretáveis. Recomenda-se a migração do target para Triple Barrier.

---

## 7. Gráficos Gerados

| Gráfico | Arquivo |
|---------|---------|
| ROC Curves | `ml/triple_barrier_report/roc_curves.png` |
| Calibration Curves | `ml/triple_barrier_report/calibration_curves.png` |
| Metrics Evolution | `ml/triple_barrier_report/metrics_evolution.png` |
| SHAP Comparison | `ml/triple_barrier_report/shap_comparison.png` |
| Label Distribution | `ml/triple_barrier_report/label_distribution.png` |

---

*Relatório gerado automaticamente pelo pipeline Triple Barrier Evaluation.*
*Modelo: LightGBM | Parâmetros: idênticos ao v1 | Features: idênticas ao v1*
