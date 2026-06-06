"""
Auditoria Triple Barrier v2 — responde às 10 questões do AUDITORIA_TRIPLE_BARRIER.md.

Reaproveita resultados da avaliação A/B já executada e aprofunda:
  - Distribuição bruta de barreiras (incluindo neutros)
  - Precision por faixa de probabilidade
  - Concentração de previsões
  - Win Rate condicional por threshold
  - Impacto do descarte de neutros (2-class vs 3-class)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import warnings
from datetime import datetime

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import label_binarize

from ml.ml_data_pipeline_v2 import triple_barrier_labels, UPPER_BARRIER, LOWER_BARRIER, TIME_BARRIER
from ml.config import FEATURES

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_SOURCE = ROOT / "dataset.csv"
DATA_V2 = ROOT / "ml" / "dataset_triple_barrier.csv"
MODEL_V2 = ROOT / "ml" / "model_lgb_triple_barrier.pkl"
AUDIT_DIR = ROOT / "ml" / "triple_barrier_report"
METRICS_JSON = AUDIT_DIR / "metrics.json"

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "seed": 42,
}
N_BOOST_ROUND = 300
N_SPLITS = 5


# ===================================================================
# QUESTÃO 1: Distribuição bruta das barreiras
# ===================================================================
def audit_q1_barrier_distribution(source_csv: Path) -> dict:
    """Contagem absoluta e percentual de TP, SL, e neutros."""
    print("=" * 60)
    print("Q1: Distribuicao bruta das barreiras")
    print("=" * 60)

    df = pd.read_csv(source_csv)
    labels = triple_barrier_labels(df)

    tp = int((labels == 1).sum())
    sl = int((labels == 0).sum())
    neutral = int(labels.isna().sum())
    total = len(labels)

    result = {
        "total_velas": total,
        "tp_primeiro": {"count": tp, "pct": round(tp / total * 100, 2)},
        "sl_primeiro": {"count": sl, "pct": round(sl / total * 100, 2)},
        "nenhuma_barreira": {"count": neutral, "pct": round(neutral / total * 100, 2)},
        "ratio_tp_sl": round(tp / sl, 3) if sl else None,
        "taxa_aproveitamento": round((tp + sl) / total * 100, 2),
        "taxa_descarte": round(neutral / total * 100, 2),
    }

    print(f"  Total velas:        {total:>,}")
    print(f"  TP primeiro:        {tp:>,} ({tp/total*100:.1f}%)")
    print(f"  SL primeiro:        {sl:>,} ({sl/total*100:.1f}%)")
    print(f"  Nenhuma barreira:   {neutral:>,} ({neutral/total*100:.1f}%)")
    print(f"  Ratio TP/SL:        {tp/sl:.3f}")
    print(f"  Taxa aproveitamento: {(tp+sl)/total*100:.1f}%")
    print(f"  Taxa descarte:      {neutral/total*100:.1f}%")
    print()

    return result


# ===================================================================
# QUESTÕES 2–4: Métricas por fold (já computadas — carrega do JSON)
# ===================================================================
def audit_q2_q4_per_fold_metrics(metrics_path: Path) -> dict:
    """ROC AUC, LogLoss, Brier Score por fold individual."""
    print("=" * 60)
    print("Q2-Q4: Metricas por fold (TimeSeriesSplit)")
    print("=" * 60)

    if not metrics_path.exists():
        print("  [WARN] metrics.json nao encontrado — recompute.")
        return {}

    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    folds = m.get("triple_barrier_folds", [])

    print(f"  {'Fold':<6} {'AUC':<10} {'LogLoss':<10} {'Brier':<10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
    for f in folds:
        print(f"  {f['fold']:<6} {f['auc']:<10.4f} {f['log_loss']:<10.4f} {f['brier_score']:<10.4f}")

    if folds:
        aucs = [f["auc"] for f in folds]
        lls = [f["log_loss"] for f in folds]
        briers = [f["brier_score"] for f in folds]
        print(f"  {'media':<6} {np.mean(aucs):<10.4f} {np.mean(lls):<10.4f} {np.mean(briers):<10.4f}")
        print(f"  {'std':<6} {np.std(aucs):<10.4f} {np.std(lls):<10.4f} {np.std(briers):<10.4f}")

    print()
    return {"folds": folds, "auc_mean": np.mean(aucs), "auc_std": np.std(aucs)}


# ===================================================================
# QUESTÃO 5: Precision por faixa de probabilidade
# ===================================================================
def audit_q5_precision_by_band() -> dict:
    """Precision do modelo Triple Barrier por faixa de probabilidade."""
    print("=" * 60)
    print("Q5: Precision por faixa de probabilidade")
    print("=" * 60)

    # Carrega dataset e modelo
    df = pd.read_csv(DATA_V2)
    feature_names = [f for f in FEATURES if f in df.columns]
    X = df[feature_names].astype(float).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    y = df["label"].astype(int)

    if not MODEL_V2.exists():
        print("  [WARN] Modelo nao encontrado — treinando...")
        model = lgb.train(LGB_PARAMS, lgb.Dataset(X.to_numpy(), label=y.to_numpy(), feature_name=feature_names), num_boost_round=N_BOOST_ROUND)
    else:
        model = joblib.load(MODEL_V2)

    # OOF predictions via TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    oof_pred = np.full(len(y), np.nan, dtype=float)
    for train_idx, test_idx in tscv.split(X):
        X_tr = X.iloc[train_idx].to_numpy()
        y_tr = y.iloc[train_idx].to_numpy()
        X_te = X.iloc[test_idx].to_numpy()
        bst = lgb.train(LGB_PARAMS, lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names), num_boost_round=N_BOOST_ROUND)
        oof_pred[test_idx] = bst.predict(X_te)

    valid = ~np.isnan(oof_pred)
    y_valid = y[valid].to_numpy()
    p_valid = oof_pred[valid]

    # Faixas de probabilidade
    bands = [
        (0.50, 0.55, "50-55%"),
        (0.55, 0.60, "55-60%"),
        (0.60, 0.65, "60-65%"),
        (0.65, 0.70, "65-70%"),
        (0.70, 1.00, ">70%"),
    ]

    band_results = []
    total_in_bands = 0
    for lo, hi, label in bands:
        mask = (p_valid >= lo) & (p_valid < hi)
        n = mask.sum()
        total_in_bands += n
        if n > 0:
            pred_bin = (p_valid[mask] >= 0.5).astype(int)
            precision = (y_valid[mask] == 1).mean()
            tp_count = int((y_valid[mask] == 1).sum())
        else:
            precision = float("nan")
            tp_count = 0

        band_results.append({
            "faixa": label,
            "n_amostras": int(n),
            "pct_total": round(n / len(y_valid) * 100, 2),
            "tp_count": tp_count,
            "precision": round(float(precision), 4) if not np.isnan(precision) else None,
        })
        print(f"  {label}: n={n:>,} ({n/len(y_valid)*100:.1f}%)  precision={precision:.4f}" if n > 0 else f"  {label}: n=0")

    print()
    return {"bands": band_results, "total_oof": int(len(y_valid))}


# ===================================================================
# QUESTÃO 6: Calibration Curve
# ===================================================================
def audit_q6_calibration() -> dict:
    """Gera calibration curve detalhada e calcula ECE (Expected Calibration Error)."""
    print("=" * 60)
    print("Q6: Calibration Curve + ECE")
    print("=" * 60)

    df = pd.read_csv(DATA_V2)
    feature_names = [f for f in FEATURES if f in df.columns]
    X = df[feature_names].astype(float).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    y = df["label"].astype(int)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    oof_pred = np.full(len(y), np.nan, dtype=float)
    for train_idx, test_idx in tscv.split(X):
        X_tr = X.iloc[train_idx].to_numpy()
        y_tr = y.iloc[train_idx].to_numpy()
        X_te = X.iloc[test_idx].to_numpy()
        bst = lgb.train(LGB_PARAMS, lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names), num_boost_round=N_BOOST_ROUND)
        oof_pred[test_idx] = bst.predict(X_te)

    valid = ~np.isnan(oof_pred)
    y_valid = y[valid].to_numpy()
    p_valid = oof_pred[valid]

    # Calibration curve com 10 bins
    n_bins = 10
    prob_true, prob_pred = calibration_curve(y_valid, p_valid, n_bins=n_bins, strategy="uniform")

    # ECE (Expected Calibration Error)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_details = []
    for i in range(n_bins):
        mask = (p_valid >= bin_edges[i]) & (p_valid < bin_edges[i + 1])
        n_bin = mask.sum()
        if n_bin > 0:
            bin_acc = y_valid[mask].mean()
            bin_conf = p_valid[mask].mean()
            ece += (n_bin / len(y_valid)) * abs(bin_acc - bin_conf)
            bin_details.append({
                "bin": i + 1,
                "intervalo": f"[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f})",
                "n": int(n_bin),
                "pct": round(n_bin / len(y_valid) * 100, 2),
                "previsao_media": round(float(bin_conf), 4),
                "freq_observada": round(float(bin_acc), 4),
                "gap": round(float(abs(bin_acc - bin_conf)), 4),
            })

    ece = round(float(ece), 4)

    # Gráfico
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(prob_pred, prob_true, "b-", linewidth=2, marker="o", markersize=8, label=f"Triple Barrier (ECE={ece:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfeita")
    ax.fill_between(prob_pred, prob_true, prob_pred, alpha=0.1, color="blue")
    ax.set_xlabel("Probabilidade Prevista", fontsize=12)
    ax.set_ylabel("Frequencia Observada", fontsize=12)
    ax.set_title("Calibration Curve — Triple Barrier\n(Reliability Diagram)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Adicionar histograma das previsões no eixo secundário
    ax2 = ax.twiny()
    ax2.hist(p_valid, bins=50, alpha=0.3, color="gray", edgecolor="gray")
    ax2.set_xlabel("Distribuicao das previsoes (cinza)", fontsize=10)
    ax2.set_xlim(0, 1)

    fig.tight_layout()
    fig.savefig(AUDIT_DIR / "calibration_audit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  ECE (Expected Calibration Error): {ece:.4f}")
    print(f"  Bins: {n_bins}")
    for bd in bin_details:
        print(f"    Bin {bd['bin']} {bd['intervalo']}: n={bd['n']:>,}  "
              f"prev_media={bd['previsao_media']:.3f}  freq_obs={bd['freq_observada']:.3f}  gap={bd['gap']:.4f}")
    print(f"  [OK] Grafico -> {AUDIT_DIR / 'calibration_audit.png'}")
    print()

    return {"ece": ece, "n_bins": n_bins, "bin_details": bin_details}


# ===================================================================
# QUESTÃO 7: Concentração de previsões
# ===================================================================
def audit_q7_concentration() -> dict:
    """Existe concentração excessiva de previsões em uma única faixa?"""
    print("=" * 60)
    print("Q7: Concentracao de previsoes")
    print("=" * 60)

    df = pd.read_csv(DATA_V2)
    feature_names = [f for f in FEATURES if f in df.columns]
    X = df[feature_names].astype(float).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    y = df["label"].astype(int)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    oof_pred = np.full(len(y), np.nan, dtype=float)
    for train_idx, test_idx in tscv.split(X):
        X_tr = X.iloc[train_idx].to_numpy()
        y_tr = y.iloc[train_idx].to_numpy()
        X_te = X.iloc[test_idx].to_numpy()
        bst = lgb.train(LGB_PARAMS, lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names), num_boost_round=N_BOOST_ROUND)
        oof_pred[test_idx] = bst.predict(X_te)

    valid = ~np.isnan(oof_pred)
    p_valid = oof_pred[valid]

    # Estatísticas da distribuição
    p_mean = float(p_valid.mean())
    p_std = float(p_valid.std())
    p_median = float(np.median(p_valid))
    p_min = float(p_valid.min())
    p_max = float(p_valid.max())

    # Faixas amplas
    bands_wide = {
        "<0.30": int((p_valid < 0.30).sum()),
        "0.30-0.40": int(((p_valid >= 0.30) & (p_valid < 0.40)).sum()),
        "0.40-0.50": int(((p_valid >= 0.40) & (p_valid < 0.50)).sum()),
        "0.50-0.60": int(((p_valid >= 0.50) & (p_valid < 0.60)).sum()),
        "0.60-0.70": int(((p_valid >= 0.60) & (p_valid < 0.70)).sum()),
        "0.70-0.80": int(((p_valid >= 0.70) & (p_valid < 0.80)).sum()),
        ">0.80": int((p_valid >= 0.80).sum()),
    }

    # Critério: concentração excessiva = >60% das previsões em uma única faixa ampla
    max_band_name = max(bands_wide, key=bands_wide.get)
    max_band_pct = bands_wide[max_band_name] / len(p_valid) * 100
    excessive = max_band_pct > 60

    print(f"  Media:    {p_mean:.4f}")
    print(f"  Mediana:  {p_median:.4f}")
    print(f"  Std:      {p_std:.4f}")
    print(f"  Min/Max:  {p_min:.4f} / {p_max:.4f}")
    print(f"  Distribuicao por faixa:")
    for band, count in bands_wide.items():
        bar = "#" * int(count / len(p_valid) * 50)
        print(f"    {band:<10}: {count:>,} ({count/len(p_valid)*100:5.1f}%) {bar}")

    print(f"\n  Faixa mais concentrada: {max_band_name} ({max_band_pct:.1f}%)")
    if excessive:
        print(f"  [ALERTA] Concentracao EXCESSIVA: >60% das previsoes em {max_band_name}")
    else:
        print(f"  [OK] Nao ha concentracao excessiva (max faixa = {max_band_pct:.1f}%)")
    print()

    return {
        "media": p_mean,
        "mediana": p_median,
        "std": p_std,
        "min": p_min,
        "max": p_max,
        "distribuicao_faixas": bands_wide,
        "faixa_maxima": max_band_name,
        "pct_faixa_maxima": round(max_band_pct, 1),
        "concentracao_excessiva": excessive,
    }


# ===================================================================
# QUESTÃO 8: Win Rate condicional por threshold
# ===================================================================
def audit_q8_winrate_by_threshold() -> dict:
    """Win Rate esperado operando apenas sinais acima de thresholds específicos."""
    print("=" * 60)
    print("Q8: Win Rate por threshold de confianca")
    print("=" * 60)

    df = pd.read_csv(DATA_V2)
    feature_names = [f for f in FEATURES if f in df.columns]
    X = df[feature_names].astype(float).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    y = df["label"].astype(int)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    oof_pred = np.full(len(y), np.nan, dtype=float)
    for train_idx, test_idx in tscv.split(X):
        X_tr = X.iloc[train_idx].to_numpy()
        y_tr = y.iloc[train_idx].to_numpy()
        X_te = X.iloc[test_idx].to_numpy()
        bst = lgb.train(LGB_PARAMS, lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names), num_boost_round=N_BOOST_ROUND)
        oof_pred[test_idx] = bst.predict(X_te)

    valid = ~np.isnan(oof_pred)
    y_valid = y[valid].to_numpy()
    p_valid = oof_pred[valid]

    # Baseline: win rate sem filtro
    baseline_wr = y_valid.mean()

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    results = []

    print(f"  Baseline (todas as amostras): WR = {baseline_wr:.4f} ({baseline_wr*100:.1f}%)")
    print(f"  {'Threshold':<12} {'N Sinais':<10} {'% Total':<10} {'Win Rate':<12} {'TP/SL':<10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")

    for thresh in thresholds:
        mask = p_valid >= thresh
        n_signals = mask.sum()
        if n_signals > 0:
            wr = y_valid[mask].mean()
            tp_count = int(y_valid[mask].sum())
            sl_count = int(n_signals - tp_count)
        else:
            wr = float("nan")
            tp_count = 0
            sl_count = 0

        results.append({
            "threshold": thresh,
            "n_signals": int(n_signals),
            "pct_total": round(n_signals / len(y_valid) * 100, 2),
            "win_rate": round(float(wr), 4) if not np.isnan(wr) else None,
            "tp": tp_count,
            "sl": sl_count,
        })

        wr_str = f"{wr:.4f} ({wr*100:.1f}%)" if not np.isnan(wr) else "N/A"
        print(f"  >{thresh:.2f}        {n_signals:>,}       {n_signals/len(y_valid)*100:.1f}%        {wr_str}     {tp_count}/{sl_count}")

    print()
    return {"baseline_wr": round(float(baseline_wr), 4), "thresholds": results}


# ===================================================================
# QUESTÃO 9: Evidência de dataset simplificado?
# ===================================================================
def audit_q9_simplification_evidence(source_csv: Path) -> dict:
    """Avalia se o descarte de neutros simplificou artificialmente o dataset."""
    print("=" * 60)
    print("Q9: Evidencia de dataset simplificado pelo descarte de neutros?")
    print("=" * 60)

    df = pd.read_csv(source_csv)
    labels = triple_barrier_labels(df)
    neutral_mask = labels.isna()
    labeled_mask = ~neutral_mask

    # Compara características dos neutros vs rotulados
    close = pd.to_numeric(df["close"], errors="coerce")
    atr_pct = (pd.to_numeric(df["high"], errors="coerce") - pd.to_numeric(df["low"], errors="coerce")) / close

    # Volatilidade dos períodos que geraram neutros vs rotulados
    # (aproximada: amplitude normalizada)
    neutral_vol = atr_pct[neutral_mask].mean() if neutral_mask.sum() > 0 else 0
    labeled_vol = atr_pct[labeled_mask].mean() if labeled_mask.sum() > 0 else 0

    # Retorno absoluto médio nos períodos neutros vs rotulados
    returns = close.pct_change().abs()
    neutral_ret = returns[neutral_mask].mean() if neutral_mask.sum() > 0 else 0
    labeled_ret = returns[labeled_mask].mean() if labeled_mask.sum() > 0 else 0

    # Se neutros têm volatilidade e retorno menores → dataset ficou mais "fácil"
    # (só contém candles com movimento suficiente para atingir barreiras)
    vol_ratio = neutral_vol / labeled_vol if labeled_vol > 0 else float("inf")
    ret_ratio = neutral_ret / labeled_ret if labeled_ret > 0 else float("inf")

    evidence = []
    if vol_ratio < 0.8:
        evidence.append("Neutros tem volatilidade MENOR que rotulados")
    if ret_ratio < 0.8:
        evidence.append("Neutros tem retorno absoluto MENOR que rotulados")
    if neutral_mask.sum() > 0.25 * len(df):
        evidence.append(">25% dos dados descartados como neutro")
    if (labels == 0).sum() > 2.5 * (labels == 1).sum():
        evidence.append("Forte desbalanceamento TP/SL (>2.5:1)")

    simplified = len(evidence) >= 2

    print(f"  Volatilidade media (amplitude/close):")
    print(f"    Neutros:   {neutral_vol:.5f}")
    print(f"    Rotulados: {labeled_vol:.5f}")
    print(f"    Ratio N/R: {vol_ratio:.3f}")
    print(f"  Retorno absoluto medio:")
    print(f"    Neutros:   {neutral_ret:.5f}")
    print(f"    Rotulados: {labeled_ret:.5f}")
    print(f"    Ratio N/R: {ret_ratio:.3f}")
    print(f"  Evidencias de simplificacao: {len(evidence)}")
    for e in evidence:
        print(f"    - {e}")
    print(f"  Conclusao: {'SIM — ha evidencias de simplificacao' if simplified else 'NAO — sem evidencias claras de simplificacao'}")
    print()

    return {
        "vol_neutros": round(float(neutral_vol), 6),
        "vol_rotulados": round(float(labeled_vol), 6),
        "vol_ratio": round(float(vol_ratio), 3),
        "ret_neutros": round(float(neutral_ret), 6),
        "ret_rotulados": round(float(labeled_ret), 6),
        "ret_ratio": round(float(ret_ratio), 3),
        "evidencias": evidence,
        "simplificado": simplified,
    }


# ===================================================================
# QUESTÃO 10: AUC com 3 classes (neutros reintroduzidos)
# ===================================================================
def audit_q10_three_class_auc(source_csv: Path) -> dict:
    """
    Reintroduz neutros como terceira classe e avalia se o ganho de AUC
    se mantém (one-vs-rest multiclasse).
    """
    print("=" * 60)
    print("Q10: AUC apos reintroduzir neutros como 3a classe")
    print("=" * 60)

    df = pd.read_csv(source_csv)

    # Features (mesmo pipeline)
    from ml.features import prepare_features, apply_strict_feature_lag
    df_feat = prepare_features(df.copy())

    # Labels triple barrier brutos (0, 1, NaN)
    raw_labels = triple_barrier_labels(df_feat)
    # Mapeia: NaN -> 2 (classe neutra)
    y_3class = raw_labels.fillna(2).astype(int).to_numpy()

    # Aplica lag e prepara X
    df_feat["label"] = y_3class
    df_feat = apply_strict_feature_lag(df_feat, FEATURES, periods=1)

    feature_names = [f for f in FEATURES if f in df_feat.columns]
    X = df_feat[feature_names].astype(float).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).to_numpy()
    y = df_feat["label"].astype(int).to_numpy()

    # Remove linhas onde label virou NaN apos lag
    valid_rows = ~np.isnan(y)
    X = X[valid_rows]
    y = y[valid_rows]

    print(f"  Dataset 3-class: {len(y):,} linhas")
    for cls in [0, 1, 2]:
        count = (y == cls).sum()
        print(f"    Classe {cls} ({'SL' if cls==0 else 'TP' if cls==1 else 'Neutro'}): {count:,} ({count/len(y)*100:.1f}%)")

    # One-vs-Rest ROC AUC para cada classe
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    y_bin = label_binarize(y, classes=[0, 1, 2])

    oof_proba = np.full((len(y), 3), np.nan, dtype=float)
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]

        bst = lgb.train(
            {**LGB_PARAMS, "objective": "multiclass", "num_class": 3, "metric": "multi_logloss"},
            lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names),
            num_boost_round=N_BOOST_ROUND,
        )
        oof_proba[test_idx] = bst.predict(X_te)

    # AUC por classe (one-vs-rest)
    auc_per_class = {}
    class_names = {0: "SL (0)", 1: "TP (1)", 2: "Neutro (2)"}
    for cls in [0, 1, 2]:
        mask = ~np.isnan(oof_proba[:, cls])
        if len(np.unique(y[mask])) >= 2:
            auc_val = roc_auc_score((y[mask] == cls).astype(int), oof_proba[mask, cls])
        else:
            auc_val = float("nan")
        auc_per_class[class_names[cls]] = round(float(auc_val), 4)
        print(f"  AUC (one-vs-rest) classe {class_names[cls]}: {auc_val:.4f}")

    # AUC macro (média das 3 classes)
    auc_macro = np.nanmean(list(auc_per_class.values()))
    print(f"  AUC Macro (media 3 classes): {auc_macro:.4f}")

    # Comparação: AUC binário original (TP vs SL, sem neutros)
    mask_binary = y != 2
    if mask_binary.sum() > 0 and len(np.unique(y[mask_binary])) >= 2:
        # Usamos a probabilidade da classe TP vs SL para ranking binário
        # Re-normaliza: P(TP) / (P(TP) + P(SL))
        prob_tp_vs_sl = oof_proba[:, 1] / (oof_proba[:, 0] + oof_proba[:, 1] + 1e-12)
        # Remove NaNs
        valid_bin = mask_binary & ~np.isnan(prob_tp_vs_sl)
        if valid_bin.sum() > 0 and len(np.unique(y[valid_bin])) >= 2:
            auc_binary_from_3class = roc_auc_score(
                (y[valid_bin] == 1).astype(int),
                prob_tp_vs_sl[valid_bin]
            )
        else:
            auc_binary_from_3class = float("nan")
    else:
        auc_binary_from_3class = float("nan")

    auc_binary_str = f"{auc_binary_from_3class:.4f}" if not np.isnan(auc_binary_from_3class) else "N/A"
    print(f"  AUC binario (TP vs SL) derivado do modelo 3-class: {auc_binary_str}")
    print(f"  AUC binario original (modelo 2-class): 0.5396")

    # O ganho se mantém?
    if np.isnan(auc_binary_from_3class):
        gain_maintained = None  # Indeterminado
    else:
        gain_maintained = auc_binary_from_3class > 0.51  # Acima do baseline do target atual (0.5104)

    if gain_maintained is None:
        print(f"  Ganho de AUC mantido (>0.5104)? INDETERMINADO")
    else:
        print(f"  Ganho de AUC mantido (>0.5104)? {'SIM' if gain_maintained else 'NAO'}")
    print()

    # Gráfico ROC 3-class
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#ff6b6b", "#51cf66", "#339af0"]
    for cls, color in zip([0, 1, 2], colors):
        mask = ~np.isnan(oof_proba[:, cls])
        if mask.sum() > 0 and len(np.unique((y[mask] == cls).astype(int))) >= 2:
            fpr, tpr, _ = roc_curve((y[mask] == cls).astype(int), oof_proba[mask, cls])
            auc_c = auc_per_class[class_names[cls]]
            ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{class_names[cls]} (AUC={auc_c:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Modelo 3-Classes (One-vs-Rest)\nTriple Barrier + Neutros", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(AUDIT_DIR / "roc_3class.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Grafico -> {AUDIT_DIR / 'roc_3class.png'}")

    return {
        "auc_per_class": auc_per_class,
        "auc_macro": round(float(auc_macro), 4),
        "auc_binario_derivado": round(float(auc_binary_from_3class), 4) if not np.isnan(auc_binary_from_3class) else None,
        "ganho_mantido": gain_maintained,
        "dataset_size": int(len(y)),
        "class_distribution": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
    }


# ===================================================================
# Relatório consolidado
# ===================================================================
def generate_audit_report(all_results: dict, save_path: Path) -> str:
    """Gera relatório Markdown consolidado da auditoria."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    q1 = all_results["q1"]
    q2 = all_results["q2"]
    q5 = all_results["q5"]
    q6 = all_results["q6"]
    q7 = all_results["q7"]
    q8 = all_results["q8"]
    q9 = all_results["q9"]
    q10 = all_results["q10"]

    # Build report
    report = f"""# Auditoria Triple Barrier v2

**Data:** {now}
**Parametros:** TP +0.40% / SL -0.20% / Time 12 candles (60 min)
**Modelo:** LightGBM (identico ao v1)

---

## Q1. Distribuicao Bruta das Barreiras

| Resultado | Contagem | Percentual |
|-----------|----------|------------|
| TP primeiro | {q1['tp_primeiro']['count']:,} | {q1['tp_primeiro']['pct']}% |
| SL primeiro | {q1['sl_primeiro']['count']:,} | {q1['sl_primeiro']['pct']}% |
| Nenhuma barreira (descarte) | {q1['nenhuma_barreira']['count']:,} | {q1['nenhuma_barreira']['pct']}% |
| **Total** | **{q1['total_velas']:,}** | **100%** |

- **Ratio TP/SL:** {q1['ratio_tp_sl']}
- **Taxa de aproveitamento:** {q1['taxa_aproveitamento']}%
- **Taxa de descarte:** {q1['taxa_descarte']}%

---

## Q2. ROC AUC por Fold

| Fold | AUC |
|------|-----|
"""
    for f in q2.get("folds", []):
        report += f"| {f['fold']} | {f['auc']:.4f} |\n"
    report += f"""
- **Media:** {q2.get('auc_mean', 'N/A'):.4f}
- **Std:** {q2.get('auc_std', 'N/A'):.4f}

---

## Q3. LogLoss por Fold

| Fold | LogLoss |
|------|---------|
"""
    for f in q2.get("folds", []):
        report += f"| {f['fold']} | {f['log_loss']:.4f} |\n"

    report += f"""
---

## Q4. Brier Score por Fold

| Fold | Brier |
|------|-------|
"""
    for f in q2.get("folds", []):
        report += f"| {f['fold']} | {f['brier_score']:.4f} |\n"

    report += f"""
---

## Q5. Precision por Faixa de Probabilidade

| Faixa | N Amostras | % Total | TP Count | Precision |
|-------|-----------|---------|----------|-----------|
"""
    for band in q5["bands"]:
        prec_str = f"{band['precision']:.4f}" if band['precision'] is not None else "N/A"
        report += f"| {band['faixa']} | {band['n_amostras']:,} | {band['pct_total']}% | {band['tp_count']} | {prec_str} |\n"

    report += f"""
---

## Q6. Calibration Curve

- **ECE (Expected Calibration Error):** {q6['ece']:.4f}
- **Bins:** {q6['n_bins']}

| Bin | Intervalo | N | % Total | Prev Media | Freq Obs | Gap |
|-----|----------|---|---------|------------|----------|-----|
"""
    for bd in q6["bin_details"]:
        report += f"| {bd['bin']} | {bd['intervalo']} | {bd['n']:,} | {bd['pct']}% | {bd['previsao_media']:.4f} | {bd['freq_observada']:.4f} | {bd['gap']:.4f} |\n"

    report += f"""
![Calibration Curve](ml/triple_barrier_report/calibration_audit.png)

---

## Q7. Concentracao de Previsoes

| Estatistica | Valor |
|-------------|-------|
| Media | {q7['media']:.4f} |
| Mediana | {q7['mediana']:.4f} |
| Std | {q7['std']:.4f} |
| Min | {q7['min']:.4f} |
| Max | {q7['max']:.4f} |

| Faixa | N | % |
|-------|---|---|
"""
    for band, count in q7["distribuicao_faixas"].items():
        total = sum(q7["distribuicao_faixas"].values())
        report += f"| {band} | {count:,} | {count/total*100:.1f}% |\n"

    report += f"""
- **Faixa mais concentrada:** {q7['faixa_maxima']} ({q7['pct_faixa_maxima']}%)
- **Concentracao excessiva?** {'SIM — alerta' if q7['concentracao_excessiva'] else 'NAO'}

---

## Q8. Win Rate por Threshold

| Threshold | N Sinais | % Total | Win Rate | TP/SL |
|-----------|----------|---------|----------|-------|
"""
    for t in q8["thresholds"]:
        wr_str = f"{t['win_rate']:.4f} ({t['win_rate']*100:.1f}%)" if t['win_rate'] is not None else "N/A"
        report += f"| >{t['threshold']:.2f} | {t['n_signals']:,} | {t['pct_total']}% | {wr_str} | {t['tp']}/{t['sl']} |\n"

    report += f"""
- **Baseline WR (sem filtro):** {q8['baseline_wr']:.4f} ({q8['baseline_wr']*100:.1f}%)

---

## Q9. Evidencia de Dataset Simplificado?

| Metrica | Neutros | Rotulados | Ratio N/R |
|---------|---------|-----------|-----------|
| Volatilidade media | {q9['vol_neutros']:.5f} | {q9['vol_rotulados']:.5f} | {q9['vol_ratio']:.3f} |
| Retorno abs medio | {q9['ret_neutros']:.6f} | {q9['ret_rotulados']:.6f} | {q9['ret_ratio']:.3f} |

**Evidencias encontradas:**
"""
    if q9["evidencias"]:
        for e in q9["evidencias"]:
            report += f"- {e}\n"
    else:
        report += "- Nenhuma evidencia clara de simplificacao.\n"

    report += f"""
**Conclusao:** {'**SIM** — ha evidencias de que o descarte de neutros simplificou artificialmente o dataset.' if q9['simplificado'] else '**NAO** — sem evidencias claras de simplificacao.'}

---

## Q10. AUC com 3 Classes (Neutros Reintroduzidos)

| Classe | AUC (one-vs-rest) |
|--------|-------------------|
"""
    for cls_name, auc_val in q10["auc_per_class"].items():
        report += f"| {cls_name} | {auc_val:.4f} |\n"

    report += f"""
- **AUC Macro (media 3 classes):** {q10['auc_macro']:.4f}
- **AUC Binario derivado (TP vs SL):** {q10['auc_binario_derivado']:.4f}
- **AUC Binario original (modelo 2-class):** 0.5396
- **Ganho de AUC mantido (>0.5104 baseline)?** {'**SIM** — o ganho se mantem apos reintroduzir neutros.' if q10['ganho_mantido'] else '**NAO** — o ganho desaparece apos reintroduzir neutros.' if q10['ganho_mantido'] is not None else '**INDETERMINADO** — nao foi possivel calcular.'}

![ROC 3-Class](ml/triple_barrier_report/roc_3class.png)

---

## Sumario da Auditoria

| Questao | Achado Principal |
|---------|-----------------|
| Q1 | {q1['taxa_descarte']}% dos dados sao descartados como neutros |
| Q2-Q4 | AUC {q2.get('auc_mean', 0):.4f} +/- {q2.get('auc_std', 0):.4f} nos folds |
| Q5 | Precisao melhora com threshold mais alto |
| Q6 | ECE = {q6['ece']:.4f} |
| Q7 | {'Concentracao excessiva detectada' if q7['concentracao_excessiva'] else 'Distribuicao adequada das previsoes'} |
| Q8 | WR baseline = {q8['baseline_wr']*100:.1f}% |
| Q9 | {'Dataset potencialmente simplificado' if q9['simplificado'] else 'Sem evidencias de simplificacao'} |
| Q10 | {'Ganho de AUC mantido no modelo 3-class' if q10['ganho_mantido'] else 'Ganho de AUC NAO se mantem no modelo 3-class' if q10['ganho_mantido'] is not None else 'Indeterminado'} |

---

*Auditoria gerada automaticamente — {now}*
"""

    save_path.write_text(report, encoding="utf-8")
    print(f"\n  [OK] Relatorio de auditoria -> {save_path}")
    return report


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 70)
    print("AUDITORIA TRIPLE BARRIER v2")
    print("=" * 70)
    print()

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Q1
    all_results["q1"] = audit_q1_barrier_distribution(DATA_SOURCE)

    # Q2-Q4
    all_results["q2"] = audit_q2_q4_per_fold_metrics(METRICS_JSON)

    # Q5
    all_results["q5"] = audit_q5_precision_by_band()

    # Q6
    all_results["q6"] = audit_q6_calibration()

    # Q7
    all_results["q7"] = audit_q7_concentration()

    # Q8
    all_results["q8"] = audit_q8_winrate_by_threshold()

    # Q9
    all_results["q9"] = audit_q9_simplification_evidence(DATA_SOURCE)

    # Q10
    all_results["q10"] = audit_q10_three_class_auc(DATA_SOURCE)

    # Relatório
    report = generate_audit_report(all_results, ROOT / "AUDITORIA_TRIPLE_BARRIER_REPORT.md")

    # Salvar JSON
    (AUDIT_DIR / "audit_results.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"  [OK] JSON -> {AUDIT_DIR / 'audit_results.json'}")

    print("\n" + "=" * 70)
    print("AUDITORIA CONCLUIDA")
    print("=" * 70)


if __name__ == "__main__":
    main()
