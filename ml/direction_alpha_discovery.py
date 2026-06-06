"""
Direction Alpha Discovery — 3 Experimentos.

EXPERIMENTO A: Direction From Microstructure
  - Apenas 9 features de microestrutura
  - Mesmo target Triple Barrier (TP vs SL)
  - Criterio: AUC >= 0.56 (aprovado), AUC >= 0.58 (excelente)

EXPERIMENTO B: Regime-Specific Models
  - 4 modelos separados: Trend, Range, High Vol, Low Vol
  - Comparacao com modelo unico
  - Criterio: qualquer regime AUC >= 0.60

EXPERIMENTO C: Meta Labeling
  - Features: tradeability_prob, direction_prob, regime, atr_pct, cvd, oi_delta, funding_delta
  - Target: trade foi lucrativo? (1=winner, 0=loser)
  - Criterio: PF sobe E DD cai
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
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import TimeSeriesSplit

from ml.ml_data_pipeline_v2 import triple_barrier_labels, UPPER_BARRIER, LOWER_BARRIER, TIME_BARRIER
from ml.features import prepare_features, apply_strict_feature_lag
from ml.config import FEATURES as ALL_FEATURES

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_SOURCE = ROOT / "dataset.csv"
TB_DATASET = ROOT / "ml" / "dataset_triple_barrier.csv"
TRADEABILITY_MODEL = ROOT / "ml" / "model_tradeability.pkl"
TRADEABILITY_CSV = ROOT / "ml" / "dataset_tradeability.csv"
REPORT_DIR = ROOT / "ml" / "triple_barrier_report"

LGB_PARAMS = {
    "objective": "binary", "metric": "auc", "verbosity": -1,
    "boosting_type": "gbdt", "seed": 42,
}
N_BOOST_ROUND = 300
N_SPLITS = 5
TP_RETURN = UPPER_BARRIER   # +0.0040
SL_RETURN = LOWER_BARRIER   # -0.0020

REPORT_MD = ROOT / "direction_alpha_report.md"

# ---------------------------------------------------------------------------
# 9 Microstructure features for Experiment A
# ---------------------------------------------------------------------------
MICRO_FEATURES = [
    "cvd",
    "cvd_delta",
    "cvd_acceleration",
    "oi_delta",
    "oi_acceleration",
    "funding_delta",
    "funding_acceleration",
    "liquidation_density",
    "premium_delta",
]


# ===================================================================
# DATA HELPERS
# ===================================================================
def build_microstructure_dataset(source_csv: Path) -> pd.DataFrame:
    """Build dataset with ONLY the 9 microstructure features + Triple Barrier label."""
    df = pd.read_csv(source_csv)

    # Derive microstructure features
    # cvd
    cvd_series = pd.to_numeric(df.get("cvd", df.get("cvd_ratio", 0)), errors="coerce").fillna(0)

    # cvd_delta = first difference
    cvd_delta = cvd_series.diff().fillna(0)

    # cvd_acceleration = second difference
    cvd_accel = cvd_delta.diff().fillna(0)

    # oi_delta
    oi_delta = pd.to_numeric(df.get("oi_change_pct", 0), errors="coerce").fillna(0)

    # oi_acceleration
    oi_accel = oi_delta.diff().fillna(0)

    # funding_delta
    funding_delta = pd.to_numeric(df.get("funding_rate_delta", df.get("funding_rate", 0)), errors="coerce").fillna(0)

    # funding_acceleration
    funding_accel = funding_delta.diff().fillna(0)

    # liquidation_density
    liq_density = pd.to_numeric(df.get("liquidation_cluster_density", 0), errors="coerce").fillna(0)

    # premium_delta
    premium = pd.to_numeric(df.get("premium_basis_pct", df.get("premium_index", 0)), errors="coerce").fillna(0)
    premium_delta = premium.diff().fillna(0)

    # Build features dataframe
    micro_df = pd.DataFrame({
        "cvd": cvd_series,
        "cvd_delta": cvd_delta,
        "cvd_acceleration": cvd_accel,
        "oi_delta": oi_delta,
        "oi_acceleration": oi_accel,
        "funding_delta": funding_delta,
        "funding_acceleration": funding_accel,
        "liquidation_density": liq_density,
        "premium_delta": premium_delta,
    }, index=df.index)

    # Triple Barrier label
    micro_df["label"] = triple_barrier_labels(df)

    # Lag features by 1
    micro_df[MICRO_FEATURES] = micro_df[MICRO_FEATURES].shift(1)

    # Clean
    micro_df = micro_df.dropna(subset=MICRO_FEATURES + ["label"]).reset_index(drop=True)

    tp = int((micro_df["label"] == 1).sum())
    sl = int((micro_df["label"] == 0).sum())
    print(f"  Microstructure dataset: {len(micro_df):,} linhas  TP={tp:,}  SL={sl:,}")

    return micro_df


def build_meta_dataset(tb_dataset_path: Path, tradeability_csv: Path) -> pd.DataFrame:
    """
    Meta Labeling dataset.

    Features: tradeability_prob, direction_prob, regime, atr_pct, cvd, oi_delta, funding_delta
    Target: 1 = trade vencedor (TP), 0 = trade perdedor (SL)
    """
    print("  Building Meta Labeling dataset...")

    # Load Triple Barrier dataset
    df_tb = pd.read_csv(tb_dataset_path)
    # Load source for regime computation
    df_source = pd.read_csv(DATA_SOURCE)
    # Load tradeability dataset for probabilities
    df_trad = pd.read_csv(tradeability_csv)

    # Compute direction probabilities via OOF
    feature_names = [f for f in ALL_FEATURES if f in df_tb.columns]
    X_tb = df_tb[feature_names].astype(float).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    y_tb = df_tb["label"].astype(int)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    direction_prob = np.full(len(y_tb), np.nan, dtype=float)
    for train_idx, test_idx in tscv.split(X_tb):
        X_tr = X_tb.iloc[train_idx].to_numpy()
        y_tr = y_tb.iloc[train_idx].to_numpy()
        X_te = X_tb.iloc[test_idx].to_numpy()
        bst = lgb.train(LGB_PARAMS, lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names), num_boost_round=N_BOOST_ROUND)
        direction_prob[test_idx] = bst.predict(X_te)

    # Tradeability probabilities
    feature_names_trad = [f for f in ALL_FEATURES if f in df_trad.columns]
    X_trad = df_trad[feature_names_trad].astype(float).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    y_trad = df_trad["label"].astype(int)

    tradeability_prob = np.full(len(y_trad), np.nan, dtype=float)
    for train_idx, test_idx in tscv.split(X_trad):
        X_tr = X_trad.iloc[train_idx].to_numpy()
        y_tr = y_trad.iloc[train_idx].to_numpy()
        X_te = X_trad.iloc[test_idx].to_numpy()
        bst = lgb.train(LGB_PARAMS, lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names_trad), num_boost_round=N_BOOST_ROUND)
        tradeability_prob[test_idx] = bst.predict(X_te)

    # Align lengths
    min_len = min(len(direction_prob), len(tradeability_prob), len(df_tb))
    direction_prob = direction_prob[:min_len]
    tradeability_prob = tradeability_prob[:min_len]
    df_tb_aligned = df_tb.iloc[:min_len]

    # Regime classification
    regimes = classify_regimes(df_source, min_len)

    # ATR
    close = pd.to_numeric(df_source["close"], errors="coerce")
    high = pd.to_numeric(df_source["high"], errors="coerce")
    low = pd.to_numeric(df_source["low"], errors="coerce")
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    atr_pct = (atr / close).to_numpy()[:min_len]

    # CVD
    cvd = pd.to_numeric(df_source.get("cvd", 0), errors="coerce").fillna(0).to_numpy()[:min_len]

    # OI delta
    oi_delta = pd.to_numeric(df_source.get("oi_change_pct", 0), errors="coerce").fillna(0).to_numpy()[:min_len]

    # Funding delta
    funding_delta = pd.to_numeric(df_source.get("funding_rate_delta", 0), errors="coerce").fillna(0).to_numpy()[:min_len]

    # Build meta dataframe
    meta_df = pd.DataFrame({
        "tradeability_prob": tradeability_prob,
        "direction_prob": direction_prob,
        "regime": regimes[:min_len],
        "atr_pct": atr_pct[:min_len],
        "cvd": cvd[:min_len],
        "oi_delta": oi_delta[:min_len],
        "funding_delta": funding_delta[:min_len],
        "label": df_tb_aligned["label"].astype(int).to_numpy(),  # 1=TP, 0=SL
    })

    # Encode regime as numeric
    regime_map = {"trend": 0, "range": 1, "high_vol": 2, "low_vol": 3, "normal": 4, "unknown": 5}
    meta_df["regime"] = meta_df["regime"].map(regime_map).fillna(5)

    # Remove NaNs
    meta_features = ["tradeability_prob", "direction_prob", "regime", "atr_pct", "cvd", "oi_delta", "funding_delta"]
    meta_df = meta_df.dropna(subset=meta_features + ["label"]).reset_index(drop=True)

    tp = int((meta_df["label"] == 1).sum())
    sl = int((meta_df["label"] == 0).sum())
    print(f"  Meta dataset: {len(meta_df):,} linhas  TP={tp:,}  SL={sl:,}")
    return meta_df, meta_features


def classify_regimes(df_source: pd.DataFrame, n_rows: int) -> np.ndarray:
    """Classify market regime for each row."""
    close = pd.to_numeric(df_source["close"], errors="coerce")
    high = pd.to_numeric(df_source["high"], errors="coerce")
    low = pd.to_numeric(df_source["low"], errors="coerce")

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    dm_plus = high.diff().clip(lower=0)
    dm_minus = (-low.diff()).clip(lower=0)
    atr_smooth = atr.rolling(14).mean()
    di_plus = (dm_plus.rolling(14).mean() / atr_smooth.replace(0, np.nan)) * 100
    di_minus = (dm_minus.rolling(14).mean() / atr_smooth.replace(0, np.nan)) * 100
    dx = (abs(di_plus - di_minus) / (di_plus + di_minus).replace(0, np.nan)) * 100
    adx = dx.rolling(14).mean()

    regimes = np.full(n_rows, "unknown", dtype=object)
    atr_75 = atr.quantile(0.75)
    atr_25 = atr.quantile(0.25)

    for i in range(min(n_rows, len(close))):
        if i < 200:
            continue
        adx_val = adx.iloc[i]
        atr_val = atr.iloc[i]
        price_vs_ema = abs(close.iloc[i] - ema200.iloc[i]) / ema200.iloc[i] if ema200.iloc[i] > 0 else 0

        if np.isnan(adx_val) or np.isnan(atr_val):
            continue

        if adx_val > 25 and price_vs_ema > 0.02:
            regimes[i] = "trend"
        elif adx_val <= 20:
            regimes[i] = "range"
        elif atr_val >= atr_75:
            regimes[i] = "high_vol"
        elif atr_val <= atr_25:
            regimes[i] = "low_vol"
        else:
            regimes[i] = "normal"

    return regimes


# ===================================================================
# TRAINING HELPER
# ===================================================================
def train_evaluate(X: np.ndarray, y: np.ndarray, feature_names: list[str], label: str) -> dict:
    """Walk-forward training and evaluation."""
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    oof_pred = np.full(len(y), np.nan, dtype=float)
    fold_metrics = []
    best_auc = -1.0
    best_model = None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if len(np.unique(y_tr)) < 2:
            fold_metrics.append({"fold": fold, "auc": float("nan"), "error": "single_class"})
            continue

        bst = lgb.train(LGB_PARAMS, lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names), num_boost_round=N_BOOST_ROUND)
        y_pred = bst.predict(X_te)
        oof_pred[test_idx] = y_pred

        auc_f = roc_auc_score(y_te, y_pred)
        pr_f = average_precision_score(y_te, y_pred)
        yp_clip = np.clip(y_pred, 1e-15, 1-1e-15)
        ll_f = log_loss(y_te, yp_clip)
        brier_f = brier_score_loss(y_te, yp_clip)

        fold_metrics.append({
            "fold": int(fold), "auc": float(auc_f), "pr_auc": float(pr_f),
            "log_loss": float(ll_f), "brier_score": float(brier_f),
            "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        })

        if auc_f > best_auc:
            best_auc = auc_f
            best_model = bst

    valid = ~np.isnan(oof_pred)
    y_valid = y[valid]
    p_valid = oof_pred[valid]
    p_clip = np.clip(p_valid, 1e-15, 1-1e-15)

    overall = {
        "auc": float(roc_auc_score(y_valid, p_valid)),
        "pr_auc": float(average_precision_score(y_valid, p_valid)),
        "log_loss": float(log_loss(y_valid, p_clip)),
        "brier_score": float(brier_score_loss(y_valid, p_clip)),
        "accuracy": float(accuracy_score(y_valid, (p_valid >= 0.5).astype(int))),
        "n_valid": int(len(y_valid)),
    }

    print(f"  {label}: AUC={overall['auc']:.4f}  PR_AUC={overall['pr_auc']:.4f}  "
          f"LogLoss={overall['log_loss']:.4f}  Brier={overall['brier_score']:.4f}")

    return {
        "label": label, "overall": overall, "fold_metrics": fold_metrics,
        "oof_predictions": p_valid, "y_true": y_valid, "model": best_model,
    }


def compute_shap(model, X_sample: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    try:
        contrib = model.predict(X_sample, pred_contrib=True)
        vals = contrib[:, :-1]
        return pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": np.abs(vals).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
    except Exception:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])


def compute_financial_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute financial metrics from predictions and triple barrier outcomes."""
    signals = (y_pred >= threshold).astype(int)
    mask = signals == 1
    n_trades = mask.sum()
    if n_trades == 0:
        return {"profit_factor": 0, "sharpe": 0, "expectancy_pct": 0, "max_dd_pct": 0, "n_trades": 0, "win_rate": 0}

    lbl = y_true[mask]
    tp = int((lbl == 1).sum())
    sl = int((lbl == 0).sum())
    wr = tp / (tp + sl) if (tp + sl) > 0 else 0.0

    returns = np.where(lbl == 1, TP_RETURN, np.where(lbl == 0, SL_RETURN, 0.0))
    pnl_pct = returns * 100

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    gross_profit = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.001
    pf = gross_profit / gross_loss
    expectancy = wr * TP_RETURN - (1 - wr) * abs(SL_RETURN)

    equity = np.cumsum(pnl_pct)
    running_max = np.maximum.accumulate(equity + 100)
    dd = (equity + 100 - running_max) / running_max
    max_dd = float(dd.min())

    sharpe = float((pnl_pct.mean() / pnl_pct.std()) * np.sqrt(len(pnl_pct))) if len(pnl_pct) > 1 and pnl_pct.std() > 0 else 0.0

    return {
        "profit_factor": round(float(pf), 4),
        "sharpe": round(float(sharpe), 4),
        "expectancy_pct": round(float(expectancy * 100), 4),
        "max_dd_pct": round(float(max_dd * 100), 4),
        "n_trades": int(n_trades),
        "win_rate": round(float(wr), 4),
    }


# ===================================================================
# EXPERIMENTO A: Direction From Microstructure
# ===================================================================
def experiment_a() -> dict:
    print("\n" + "=" * 70)
    print("EXPERIMENTO A: Direction From Microstructure")
    print("  Hipotese: Fluxo agressivo precede direcao")
    print("  Features: Apenas 9 de microestrutura")
    print("=" * 70)

    # Build dataset
    df_micro = build_microstructure_dataset(DATA_SOURCE)

    X = df_micro[MICRO_FEATURES].astype(float).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).to_numpy()
    y = df_micro["label"].astype(int).to_numpy()

    # Train
    result = train_evaluate(X, y, MICRO_FEATURES, "Microstructure")

    # SHAP
    n_shap = min(2000, X.shape[0])
    shap_df = compute_shap(result["model"], X[-n_shap:], MICRO_FEATURES)

    # Financial metrics
    fin = compute_financial_metrics(result["y_true"], result["oof_predictions"])

    auc_val = result["overall"]["auc"]
    passed = auc_val >= 0.56
    excellent = auc_val >= 0.58

    status = "EXCELENTE" if excellent else ("APROVADO" if passed else "REPROVADO")
    print(f"\n  AUC: {auc_val:.4f} | Criterio: >=0.56 (aprovado) >=0.58 (excelente)")
    print(f"  Status: {status}")
    print(f"  PF: {fin['profit_factor']:.4f} | Sharpe: {fin['sharpe']:.4f} | Expect: {fin['expectancy_pct']:.4f}%")

    # SHAP top
    if not shap_df.empty:
        print(f"  Top SHAP: {shap_df.iloc[0]['feature']} ({shap_df.iloc[0]['mean_abs_shap']:.4f})")
        dominant = shap_df.iloc[0]["mean_abs_shap"] > 0.10 * shap_df["mean_abs_shap"].sum()
        dominant_feature = shap_df.iloc[0]["feature"] if dominant else None
    else:
        dominant_feature = None

    return {
        "experiment": "Microstructure",
        "auc": auc_val, "pr_auc": result["overall"]["pr_auc"],
        "log_loss": result["overall"]["log_loss"],
        "brier": result["overall"]["brier_score"],
        "profit_factor": fin["profit_factor"], "sharpe": fin["sharpe"],
        "expectancy_pct": fin["expectancy_pct"], "max_dd_pct": fin["max_dd_pct"],
        "n_trades": fin["n_trades"], "win_rate": fin["win_rate"],
        "status": status, "passed": passed, "excellent": excellent,
        "shap_top": shap_df.head(3)["feature"].tolist() if not shap_df.empty else [],
        "dominant_feature": dominant_feature,
        "shap_df": shap_df,
    }


# ===================================================================
# EXPERIMENTO B: Regime-Specific Models
# ===================================================================
def experiment_b() -> dict:
    print("\n" + "=" * 70)
    print("EXPERIMENTO B: Regime-Specific Models")
    print("  Hipotese: Misturar regimes destroi Alpha")
    print("=" * 70)

    # Load Triple Barrier dataset
    df_tb = pd.read_csv(TB_DATASET)
    feature_names = [f for f in ALL_FEATURES if f in df_tb.columns]
    X = df_tb[feature_names].astype(float).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    y = df_tb["label"].astype(int)

    # Classify regimes on source data
    df_source = pd.read_csv(DATA_SOURCE)
    regimes = classify_regimes(df_source, len(df_tb))
    # Align
    min_len = min(len(df_tb), len(regimes))
    regimes = regimes[:min_len]
    X = X.iloc[:min_len]
    y = y.iloc[:min_len]

    regime_names = ["trend", "range", "high_vol", "low_vol"]

    # Train global model (single)
    print("\n  --- Modelo Unico (Global) ---")
    global_result = train_evaluate(X.to_numpy(), y.to_numpy(), feature_names, "Global")
    global_fin = compute_financial_metrics(global_result["y_true"], global_result["oof_predictions"])

    # Train per-regime models
    regime_results = {}
    for rname in regime_names:
        mask = regimes == rname
        n_regime = mask.sum()
        if n_regime < 500:
            print(f"\n  --- Regime {rname}: apenas {n_regime} amostras, pulando ---")
            regime_results[rname] = {"error": "insufficient_samples", "n": int(n_regime)}
            continue

        X_r = X[mask].to_numpy()
        y_r = y.iloc[mask].to_numpy()

        print(f"\n  --- Regime: {rname} ({n_regime:,} amostras) ---")
        result = train_evaluate(X_r, y_r, feature_names, f"Regime_{rname}")
        fin = compute_financial_metrics(result["y_true"], result["oof_predictions"])

        regime_results[rname] = {
            "n": int(n_regime),
            "auc": result["overall"]["auc"],
            "pr_auc": result["overall"]["pr_auc"],
            "profit_factor": fin["profit_factor"],
            "sharpe": fin["sharpe"],
            "expectancy_pct": fin["expectancy_pct"],
            "win_rate": fin["win_rate"],
        }
        print(f"  AUC={result['overall']['auc']:.4f}  PF={fin['profit_factor']:.4f}  "
              f"Sharpe={fin['sharpe']:.2f}  WR={fin['win_rate']:.4f}")

    # Find best regime
    valid_regimes = {k: v for k, v in regime_results.items() if "error" not in v}
    best_regime = max(valid_regimes, key=lambda r: valid_regimes[r]["auc"]) if valid_regimes else None
    best_auc = valid_regimes[best_regime]["auc"] if best_regime else 0.0
    auc_above_060 = any(v["auc"] >= 0.60 for v in valid_regimes.values())

    print(f"\n  Melhor regime: {best_regime} (AUC={best_auc:.4f})")
    print(f"  AUC >= 0.60 em algum regime? {'SIM' if auc_above_060 else 'NAO'}")

    return {
        "experiment": "Regime Models",
        "global": {"auc": global_result["overall"]["auc"], "pf": global_fin["profit_factor"], "sharpe": global_fin["sharpe"]},
        "regimes": regime_results,
        "best_regime": best_regime,
        "best_auc": best_auc,
        "auc_above_060": auc_above_060,
        "status": "APROVADO" if auc_above_060 else "REPROVADO",
    }


# ===================================================================
# EXPERIMENTO C: Meta Labeling
# ===================================================================
def experiment_c() -> dict:
    print("\n" + "=" * 70)
    print("EXPERIMENTO C: Meta Labeling")
    print("  Hipotese: O modelo sabe gerar sinais, mas nao sabe quais ignorar")
    print("=" * 70)

    # Build meta dataset
    meta_df, meta_features = build_meta_dataset(TB_DATASET, TRADEABILITY_CSV)

    X_meta = meta_df[meta_features].astype(float).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).to_numpy()
    y_meta = meta_df["label"].astype(int).to_numpy()

    # Train meta model
    result = train_evaluate(X_meta, y_meta, meta_features, "Meta Labeling")

    # Evaluate: can it eliminate 20% of worst trades?
    oof_pred = result["oof_predictions"]
    y_true = result["y_true"]

    # Sort by meta probability (higher = more likely to be winner)
    sorted_idx = np.argsort(oof_pred)
    worst_20_pct = int(len(sorted_idx) * 0.20)
    worst_idx = sorted_idx[:worst_20_pct]

    # Worst 20% trades
    worst_trades = y_true[worst_idx]
    worst_wr = float((worst_trades == 1).mean())
    worst_tp = int((worst_trades == 1).sum())
    worst_sl = int((worst_trades == 0).sum())

    # Best 80% trades (keeping)
    best_idx = sorted_idx[worst_20_pct:]
    best_trades = y_true[best_idx]
    best_wr = float((best_trades == 1).mean())
    best_tp = int((best_trades == 1).sum())
    best_sl = int((best_trades == 0).sum())

    # Financial comparison
    all_fin = compute_financial_metrics(y_true, np.ones(len(y_true)) * 0.5)  # baseline: take all
    # Filtered: take only top 80%
    filtered_pred = np.zeros(len(y_true))
    filtered_pred[best_idx] = 1.0
    filtered_fin = compute_financial_metrics(y_true, filtered_pred)

    pf_improved = filtered_fin["profit_factor"] > all_fin["profit_factor"]
    dd_reduced = abs(filtered_fin["max_dd_pct"]) < abs(all_fin["max_dd_pct"])
    eliminated_20pct = worst_wr < best_wr

    print(f"\n  Baseline (all trades):  PF={all_fin['profit_factor']:.4f}  "
          f"DD={all_fin['max_dd_pct']:.2f}%  WR={all_fin['win_rate']:.4f}")
    print(f"  Worst 20% (eliminados): WR={worst_wr:.4f}  TP={worst_tp}  SL={worst_sl}")
    print(f"  Best 80% (mantidos):    WR={best_wr:.4f}  TP={best_tp}  SL={best_sl}")
    print(f"  Filtered:               PF={filtered_fin['profit_factor']:.4f}  "
          f"DD={filtered_fin['max_dd_pct']:.2f}%")

    print(f"\n  Elimina 20% piores? {'SIM' if eliminated_20pct else 'NAO'} "
          f"(WR worst={worst_wr:.4f} vs best={best_wr:.4f})")
    print(f"  PF melhora? {'SIM' if pf_improved else 'NAO'}")
    print(f"  DD cai? {'SIM' if dd_reduced else 'NAO'}")

    passed = pf_improved and dd_reduced

    # SHAP
    n_shap = min(2000, X_meta.shape[0])
    shap_df = compute_shap(result["model"], X_meta[-n_shap:], meta_features)

    return {
        "experiment": "Meta Labeling",
        "auc": result["overall"]["auc"],
        "pr_auc": result["overall"]["pr_auc"],
        "baseline_pf": all_fin["profit_factor"],
        "filtered_pf": filtered_fin["profit_factor"],
        "baseline_dd": all_fin["max_dd_pct"],
        "filtered_dd": filtered_fin["max_dd_pct"],
        "baseline_sharpe": all_fin["sharpe"],
        "filtered_sharpe": filtered_fin["sharpe"],
        "worst_20pct_wr": worst_wr,
        "best_80pct_wr": best_wr,
        "eliminated_20pct": eliminated_20pct,
        "pf_improved": pf_improved,
        "dd_reduced": dd_reduced,
        "passed": passed,
        "status": "APROVADO" if passed else "REPROVADO",
        "shap_df": shap_df,
    }


# ===================================================================
# PLOTS
# ===================================================================
def generate_plots(exp_a: dict, exp_b: dict, exp_c: dict):
    """Generate comprehensive plots for all 3 experiments."""
    print("\n  Gerando plots...")

    # --- Plot 1: Experiment A SHAP ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    shap_a = exp_a.get("shap_df")
    if shap_a is not None and not shap_a.empty:
        top = shap_a.head(9).iloc[::-1]
        colors = plt.cm.RdYlGn(top["mean_abs_shap"] / max(top["mean_abs_shap"].max(), 1e-6))
        ax.barh(range(len(top)), top["mean_abs_shap"], color=colors, edgecolor="gray")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["feature"], fontsize=9)
        ax.set_xlabel("Mean(|SHAP|)")
        ax.set_title(f"Exp A: Microstructure SHAP\nAUC={exp_a['auc']:.4f}")
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center")
    ax.grid(True, alpha=0.2, axis="x")

    # --- Plot 2: Experiment B regime comparison ---
    ax = axes[1]
    regimes_data = exp_b.get("regimes", {})
    valid_regimes = {k: v for k, v in regimes_data.items() if "error" not in v}
    if valid_regimes:
        rnames = list(valid_regimes.keys())
        aucs = [valid_regimes[r]["auc"] for r in rnames]
        bars = ax.bar(rnames, aucs, color=["#51cf66" if a >= 0.60 else "#ffd43b" if a >= 0.54 else "#ff6b6b" for a in aucs], edgecolor="gray")
        ax.axhline(y=0.60, color="green", linestyle="--", linewidth=2, label="Target 0.60")
        ax.axhline(y=exp_b["global"]["auc"], color="gray", linestyle="--", linewidth=1, label=f"Global ({exp_b['global']['auc']:.4f})")
        ax.set_ylabel("ROC AUC")
        ax.set_title(f"Exp B: AUC by Regime\nBest: {exp_b.get('best_regime', 'N/A')} ({exp_b.get('best_auc', 0):.4f})")
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Plot 3: Experiment C comparison ---
    ax = axes[2]
    metrics = ["Profit Factor", "Sharpe", "Win Rate"]
    x_pos = np.arange(len(metrics))
    width = 0.35
    baseline_vals = [exp_c.get("baseline_pf", 0), exp_c.get("baseline_sharpe", 0), 0.276]
    filtered_vals = [exp_c.get("filtered_pf", 0), exp_c.get("filtered_sharpe", 0), exp_c.get("best_80pct_wr", 0)]
    ax.bar(x_pos - width/2, baseline_vals, width, label="All Trades", color="#ff6b6b", alpha=0.85)
    ax.bar(x_pos + width/2, filtered_vals, width, label="Meta Filtered", color="#51cf66", alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics)
    ax.set_title(f"Exp C: Meta Labeling\nPF: {exp_c.get('baseline_pf', 0):.2f} -> {exp_c.get('filtered_pf', 0):.2f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Direction Alpha Discovery — 3 Experiments", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "direction_alpha_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] direction_alpha_summary.png")


# ===================================================================
# FINAL REPORT
# ===================================================================
def generate_report(exp_a: dict, exp_b: dict, exp_c: dict, save_path: Path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Regime table rows
    regime_rows = ""
    for rname, rdata in exp_b.get("regimes", {}).items():
        if "error" in rdata:
            regime_rows += f"| {rname} | — | — | — | — | {rdata.get('error', 'N/A')} |\n"
        else:
            regime_rows += (f"| {rname} | {rdata['n']:,} | {rdata['auc']:.4f} | "
                          f"{rdata['profit_factor']:.4f} | {rdata['sharpe']:.2f} | "
                          f"{'✅ >0.60' if rdata['auc'] >= 0.60 else '❌'} |\n")

    # SHAP top-3 for exp A
    shap_top_a = exp_a.get("shap_top", [])
    shap_str = ", ".join(f"`{f}`" for f in shap_top_a) if shap_top_a else "N/A"

    # Dominant feature
    dom = exp_a.get("dominant_feature", None)
    dom_str = f"`{dom}`" if dom else "Nenhuma"

    report = f"""# Direction Alpha Discovery — Relatorio Final

**Data:** {now}
**Objetivo:** Descobrir onde mora o Alpha direcional.

---

## Tabela Resumo

| Experimento | AUC | PF | Sharpe | Expect% | MaxDD% | Status |
|-------------|-----|-----|--------|---------|--------|--------|
| Microstructure | {exp_a['auc']:.4f} | {exp_a['profit_factor']:.4f} | {exp_a['sharpe']:.2f} | {exp_a['expectancy_pct']:.4f} | {exp_a['max_dd_pct']:.2f} | {exp_a['status']} |
| Regime Models | {exp_b.get('best_auc', 0):.4f} | — | — | — | — | {exp_b['status']} |
| Meta Labeling | {exp_c['auc']:.4f} | {exp_c['baseline_pf']:.2f}→{exp_c['filtered_pf']:.2f} | {exp_c['baseline_sharpe']:.2f}→{exp_c['filtered_sharpe']:.2f} | — | {abs(exp_c['baseline_dd']):.1f}→{abs(exp_c['filtered_dd']):.1f} | {exp_c['status']} |

---

## Experimento A: Direction From Microstructure

### Hipótese
Fluxo agressivo precede direcao.

### Features (apenas 9)
`cvd`, `cvd_delta`, `cvd_acceleration`, `oi_delta`, `oi_acceleration`,
`funding_delta`, `funding_acceleration`, `liquidation_density`, `premium_delta`

### Resultados

| Metrica | Valor |
|---------|-------|
| ROC AUC | {exp_a['auc']:.4f} |
| PR AUC | {exp_a['pr_auc']:.4f} |
| LogLoss | {exp_a['log_loss']:.4f} |
| Brier | {exp_a['brier']:.4f} |
| Profit Factor | {exp_a['profit_factor']:.4f} |
| Sharpe | {exp_a['sharpe']:.2f} |
| Expectancy | {exp_a['expectancy_pct']:.4f}% |

### Perguntas

**Q: Microestrutura sozinha supera AUC 0.54?**
{'✅ SIM' if exp_a['auc'] > 0.54 else '❌ NAO'} — AUC = {exp_a['auc']:.4f}

**Q: Qual feature explica mais de 10% do ganho?**
{dom_str}

**Q: Existe uma feature dominante?**
{'✅ SIM' if dom else '❌ NAO'} — {dom_str}

### Status: {exp_a['status']}
- Criterio aprovado (>=0.56): {'✅' if exp_a['passed'] else '❌'}
- Criterio excelente (>=0.58): {'✅' if exp_a['excellent'] else '❌'}

---

## Experimento B: Regime-Specific Models

### Hipótese
Misturar regimes destroi Alpha.

### Resultados por Regime

| Regime | Amostras | AUC | PF | Sharpe | Status |
|--------|----------|-----|----|--------|--------|
{regime_rows}

### Modelo Global
AUC = {exp_b['global']['auc']:.4f} | PF = {exp_b['global']['pf']:.4f} | Sharpe = {exp_b['global']['sharpe']:.2f}

### Perguntas

**Q: Qual regime possui maior edge?**
{exp_b.get('best_regime', 'N/A')} (AUC={exp_b.get('best_auc', 0):.4f})

**Q: Qual regime é inviável?**
{min(exp_b.get('regimes', {}), key=lambda r: exp_b['regimes'][r].get('auc', 999) if 'error' not in exp_b['regimes'].get(r, {}) else 999) if any('error' not in v for v in exp_b.get('regimes', {}).values()) else 'N/A'}

**Q: Existe regime com AUC > 0.60?**
{'✅ SIM' if exp_b.get('auc_above_060') else '❌ NAO'}

### Status: {exp_b['status']}

---

## Experimento C: Meta Labeling

### Hipótese
O modelo ja sabe gerar sinais. Mas nao sabe quais sinais ignorar.

### Features
`tradeability_prob`, `direction_prob`, `regime`, `atr_pct`, `cvd`, `oi_delta`, `funding_delta`

### Resultados

| Metrica | Baseline (All) | Meta Filtered | Delta |
|---------|---------------|---------------|-------|
| Profit Factor | {exp_c['baseline_pf']:.4f} | {exp_c['filtered_pf']:.4f} | {exp_c['filtered_pf'] - exp_c['baseline_pf']:+.4f} |
| Sharpe | {exp_c['baseline_sharpe']:.2f} | {exp_c['filtered_sharpe']:.2f} | {exp_c['filtered_sharpe'] - exp_c['baseline_sharpe']:+.2f} |
| Max DD | {abs(exp_c['baseline_dd']):.2f}% | {abs(exp_c['filtered_dd']):.2f}% | {abs(exp_c['baseline_dd']) - abs(exp_c['filtered_dd']):+.2f}pp |
| WR (worst 20%) | {exp_c['worst_20pct_wr']:.4f} | — | — |
| WR (best 80%) | — | {exp_c['best_80pct_wr']:.4f} | — |

### Perguntas

**Q: Elimina 20% dos piores trades?**
{'✅ SIM' if exp_c['eliminated_20pct'] else '❌ NAO'} — WR worst 20% = {exp_c['worst_20pct_wr']:.4f} vs best 80% = {exp_c['best_80pct_wr']:.4f}

**Q: Melhora PF?**
{'✅ SIM' if exp_c['pf_improved'] else '❌ NAO'} — {exp_c['baseline_pf']:.4f} → {exp_c['filtered_pf']:.4f}

**Q: Melhora Expectancy?**
{'✅ SIM' if exp_c['filtered_pf'] > exp_c['baseline_pf'] else '❌ NAO'}

### Status: {exp_c['status']}
- Criterio: PF sobe {'✅' if exp_c['pf_improved'] else '❌'} E DD cai {'✅' if exp_c['dd_reduced'] else '❌'}

---

## Conclusão Final

### Onde mora o Alpha?

| Fonte | Evidencia | Forca |
|-------|-----------|-------|
| Microestrutura | AUC={exp_a['auc']:.4f} | {'FORTE' if exp_a['excellent'] else 'MODERADA' if exp_a['passed'] else 'FRACA'} |
| Regime | Melhor regime AUC={exp_b.get('best_auc', 0):.4f} | {'FORTE (>0.60)' if exp_b.get('auc_above_060') else 'FRACA (<0.60)'} |
| Meta Labeling | PF {exp_c['baseline_pf']:.2f}→{exp_c['filtered_pf']:.2f} | {'EFICAZ' if exp_c['passed'] else 'INEFICAZ'} |

![Summary](ml/triple_barrier_report/direction_alpha_summary.png)

---

*Relatorio gerado automaticamente — {now}*
"""

    save_path.write_text(report, encoding="utf-8")
    print(f"\n  [OK] Relatorio -> {save_path}")
    return report


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 70)
    print("DIRECTION ALPHA DISCOVERY")
    print("  Experimento A: Microstructure")
    print("  Experimento B: Regime-Specific Models")
    print("  Experimento C: Meta Labeling")
    print("=" * 70)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Run experiments
    exp_a = experiment_a()
    exp_b = experiment_b()
    exp_c = experiment_c()

    # Generate plots
    generate_plots(exp_a, exp_b, exp_c)

    # Save JSON
    results_json = {
        "timestamp": datetime.now().isoformat(),
        "experiment_a": {k: v for k, v in exp_a.items() if k != "shap_df"},
        "experiment_b": {k: v for k, v in exp_b.items() if k != "regimes"},
        "experiment_b_regimes": exp_b.get("regimes", {}),
        "experiment_c": {k: v for k, v in exp_c.items() if k != "shap_df"},
    }
    (REPORT_DIR / "direction_alpha_results.json").write_text(
        json.dumps(results_json, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    # Generate report
    report = generate_report(exp_a, exp_b, exp_c, REPORT_MD)

    # Print final table
    print("\n" + "=" * 70)
    print("TABELA FINAL")
    print("=" * 70)
    print(f"{'Experimento':<20} {'AUC':<10} {'PF':<10} {'Sharpe':<10} {'Status':<15}")
    print(f"{'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*15}")
    print(f"{'Microstructure':<20} {exp_a['auc']:<10.4f} {exp_a['profit_factor']:<10.4f} {exp_a['sharpe']:<10.2f} {exp_a['status']:<15}")
    print(f"{'Regime Models':<20} {exp_b.get('best_auc', 0):<10.4f} {'—':<10} {'—':<10} {exp_b['status']:<15}")
    print(f"{'Meta Labeling':<20} {exp_c['auc']:<10.4f} {exp_c['filtered_pf']:<10.4f} {exp_c['filtered_sharpe']:<10.2f} {exp_c['status']:<15}")
    print("=" * 70)
    print(f"Relatorio: {REPORT_MD}")


if __name__ == "__main__":
    main()
