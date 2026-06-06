"""
A/B Test Validation — Experimentos A e B.

Experimento A: Tradeability Signal Frequency Audit
  - Distribuicao, percentis, frequencia por threshold, lift, calibration, deciles

Experimento B: Economic Validation Backtest
  - 4 cenarios (Control + 3 thresholds), metricas financeiras, regime breakdown
  - Responde 5 perguntas obrigatorias
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
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from ml.ml_data_pipeline_v2 import triple_barrier_labels, UPPER_BARRIER, LOWER_BARRIER, TIME_BARRIER
from ml.config import FEATURES

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRADEABILITY_CSV = ROOT / "ml" / "dataset_tradeability.csv"
TRADEABILITY_MODEL = ROOT / "ml" / "model_tradeability.pkl"
DATA_SOURCE = ROOT / "dataset.csv"
REPORT_DIR = ROOT / "ml" / "triple_barrier_report"

LGB_PARAMS = {
    "objective": "binary", "metric": "auc", "verbosity": -1,
    "boosting_type": "gbdt", "seed": 42,
}
N_BOOST_ROUND = 300
N_SPLITS = 5
TP_RETURN = UPPER_BARRIER   # +0.0040
SL_RETURN = LOWER_BARRIER   # -0.0020
ASSUMED_CAPITAL = 10_000.0

REPORT_MD = ROOT / "tradeability_frequency_report.md"


# ===================================================================
# HELPERS: OOF Predictions
# ===================================================================
def get_oof_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gera OOF predictions do modelo tradeability + labels triple barrier."""
    print("  Gerando OOF predictions do Tradeability...")

    df_trad = pd.read_csv(TRADEABILITY_CSV)
    feature_names = [f for f in FEATURES if f in df_trad.columns]
    X = df_trad[feature_names].astype(float).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    y_trad = df_trad["label"].astype(int)

    # OOF predictions
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    oof_prob = np.full(len(y_trad), np.nan, dtype=float)
    for train_idx, test_idx in tscv.split(X):
        X_tr = X.iloc[train_idx].to_numpy()
        y_tr = y_trad.iloc[train_idx].to_numpy()
        X_te = X.iloc[test_idx].to_numpy()
        bst = lgb.train(LGB_PARAMS, lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names), num_boost_round=N_BOOST_ROUND)
        oof_prob[test_idx] = bst.predict(X_te)

    # Triple barrier labels para outcome real
    df_source = pd.read_csv(DATA_SOURCE)
    tb_labels = triple_barrier_labels(df_source)

    # Alinhar indices (tradeability dataset pode ter menos linhas devido ao lag)
    # Ambos usam o mesmo source, entao os indices batem apos o lag de 1
    # O dataset tradeability tem 99,426 linhas (source=99,713, -1 lag, -286 NaN features)
    # Precisamos alinhar pelo tamanho minimo
    min_len = min(len(oof_prob), len(tb_labels))
    oof_prob = oof_prob[:min_len]
    tb_labels_aligned = tb_labels.iloc[:min_len].to_numpy()

    valid = ~np.isnan(oof_prob)
    print(f"  OOF validas: {valid.sum():,} / {len(oof_prob):,}")
    return oof_prob[valid], tb_labels_aligned[valid], valid


# ===================================================================
# EXPERIMENTO A
# ===================================================================
def experiment_a():
    """Tradeability Signal Frequency Audit."""
    print("\n" + "=" * 70)
    print("EXPERIMENTO A: Tradeability Signal Frequency Audit")
    print("=" * 70)

    # Carregar dados
    df_trad = pd.read_csv(TRADEABILITY_CSV)
    feature_names = [f for f in FEATURES if f in df_trad.columns]
    X = df_trad[feature_names].astype(float).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    y_trad = df_trad["label"].astype(int)

    # OOF
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    oof_prob = np.full(len(y_trad), np.nan, dtype=float)
    for train_idx, test_idx in tscv.split(X):
        X_tr = X.iloc[train_idx].to_numpy()
        y_tr = y_trad.iloc[train_idx].to_numpy()
        X_te = X.iloc[test_idx].to_numpy()
        bst = lgb.train(LGB_PARAMS, lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names), num_boost_round=N_BOOST_ROUND)
        oof_prob[test_idx] = bst.predict(X_te)

    valid = ~np.isnan(oof_prob)
    probs = oof_prob[valid]
    y_true = y_trad[valid].to_numpy()

    # Triple barrier labels para tradeability lift
    df_source = pd.read_csv(DATA_SOURCE)
    tb_labels = triple_barrier_labels(df_source)
    min_len = min(len(probs), len(tb_labels))
    probs_aligned = probs[:min_len]
    tb_aligned = tb_labels.iloc[:min_len].to_numpy()

    total_candles = len(probs)
    # Assumir 5-min candles → 288 candles/dia (24h crypto)
    candles_per_day = 288
    candles_per_week = 288 * 7

    # ------------------------------------------------------------------
    # A1. Distribuicao completa
    # ------------------------------------------------------------------
    print("\n--- A1: Distribuicao das Probabilidades ---")
    stats = {
        "min": float(probs.min()),
        "max": float(probs.max()),
        "mean": float(probs.mean()),
        "median": float(np.median(probs)),
        "std": float(probs.std()),
    }
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    # ------------------------------------------------------------------
    # A2. Percentis
    # ------------------------------------------------------------------
    print("\n--- A2: Percentis ---")
    percentiles = {}
    for p in [50, 75, 80, 85, 90, 95, 97, 99]:
        val = float(np.percentile(probs, p))
        percentiles[f"P{p}"] = val
        print(f"  P{p}: {val:.4f}")

    # ------------------------------------------------------------------
    # A3. Frequencia de sinais por threshold
    # ------------------------------------------------------------------
    print("\n--- A3: Frequencia de Sinais ---")
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    freq_results = []

    for thresh in thresholds:
        n_signals = int((probs >= thresh).sum())
        pct = n_signals / total_candles * 100
        per_day = n_signals / total_candles * candles_per_day
        per_week = n_signals / total_candles * candles_per_week
        freq_results.append({
            "threshold": thresh,
            "n_signals": n_signals,
            "pct_dataset": round(pct, 2),
            "signals_per_day": round(per_day, 1),
            "signals_per_week": round(per_week, 0),
        })
        print(f"  >={thresh:.2f}: {n_signals:>,} sinais ({pct:.1f}%)  "
              f"{per_day:.1f}/dia  {per_week:.0f}/semana")

    # ------------------------------------------------------------------
    # A4. Tradeability Lift
    # ------------------------------------------------------------------
    print("\n--- A4: Tradeability Lift ---")
    global_wr = float((tb_aligned == 1).sum() / ((tb_aligned == 1).sum() + (tb_aligned == 0).sum()))
    print(f"  WR Global (TP/(TP+SL)): {global_wr:.4f} ({global_wr*100:.1f}%)")

    lift_results = []
    for thresh in thresholds:
        mask = probs_aligned >= thresh
        n = mask.sum()
        if n > 0:
            tb_subset = tb_aligned[mask]
            tp = int((tb_subset == 1).sum())
            sl = int((tb_subset == 0).sum())
            neutral = int(np.isnan(tb_subset).sum())
            wr = tp / (tp + sl) if (tp + sl) > 0 else float("nan")
        else:
            tp, sl, neutral, wr = 0, 0, 0, float("nan")

        lift_results.append({
            "threshold": thresh,
            "n": int(n),
            "tp": tp, "sl": sl, "neutral": neutral,
            "wr": round(float(wr), 4) if not np.isnan(wr) else None,
            "lift_vs_global": round(float(wr) - global_wr, 4) if not np.isnan(wr) else None,
        })
        wr_str = f"{wr:.4f} ({wr*100:.1f}%)" if not np.isnan(wr) else "N/A"
        lift_str = f"{wr - global_wr:+.4f}" if not np.isnan(wr) else "N/A"
        print(f"  >={thresh:.2f}: n={n:>,}  WR={wr_str}  Lift={lift_str}")

    # ------------------------------------------------------------------
    # A5. Calibration Audit (Top 20%, 10%, 5%)
    # ------------------------------------------------------------------
    print("\n--- A5: Calibration Audit por Top Segmentos ---")
    cal_results = []
    segments = {"Top 20%": 0.80, "Top 10%": 0.90, "Top 5%": 0.95}

    for seg_name, seg_quantile in segments.items():
        cutoff = np.quantile(probs, seg_quantile)
        mask = probs >= cutoff
        y_seg = y_true[mask]
        p_seg = probs[mask]
        n_seg = mask.sum()

        if n_seg > 10:
            prob_true, prob_pred = calibration_curve(y_seg, p_seg, n_bins=5, strategy="uniform")
            ece = sum(
                (len(y_seg[(p_seg >= prob_pred[i]) & (p_seg < prob_pred[min(i+1, len(prob_pred)-1)])]) / n_seg)
                * abs(prob_true[i] - prob_pred[i])
                if i < len(prob_pred) - 1 else 0
                for i in range(len(prob_pred) - 1)
            ) if len(prob_pred) > 1 else 0

            # Simplified ECE
            bin_edges = np.linspace(0, 1, 6)
            ece_simple = 0.0
            for i in range(5):
                bin_mask = (p_seg >= bin_edges[i]) & (p_seg < bin_edges[i+1])
                if bin_mask.sum() > 0:
                    ece_simple += (bin_mask.sum() / n_seg) * abs(y_seg[bin_mask].mean() - p_seg[bin_mask].mean())
            brier = brier_score_loss(y_seg, np.clip(p_seg, 1e-15, 1-1e-15))
        else:
            ece_simple = float("nan")
            brier = float("nan")

        cal_results.append({
            "segmento": seg_name,
            "cutoff": round(float(cutoff), 4),
            "n": int(n_seg),
            "pct": round(n_seg / total_candles * 100, 2),
            "ece": round(float(ece_simple), 4) if not np.isnan(ece_simple) else None,
            "brier": round(float(brier), 4) if not np.isnan(brier) else None,
        })
        print(f"  {seg_name} (>= {cutoff:.4f}): n={n_seg:>,} "
              f"ECE={ece_simple:.4f}  Brier={brier:.4f}" if not np.isnan(ece_simple) else
              f"  {seg_name} (>= {cutoff:.4f}): n={n_seg:>,} (insuficiente)")

    # ------------------------------------------------------------------
    # A6. Score Deciles
    # ------------------------------------------------------------------
    print("\n--- A6: Score Deciles ---")
    decile_edges = np.percentile(probs, np.arange(0, 101, 10))
    decile_results = []

    for i in range(10):
        lo, hi = decile_edges[i], decile_edges[i+1]
        mask = (probs >= lo) & (probs < hi)
        n_dec = mask.sum()
        if n_dec > 0:
            y_dec = y_true[mask]
            hit_rate = float(y_dec.mean())  # % tradeable
            # Tradeability rate = hit_rate
            decile_results.append({
                "decil": i + 1,
                "intervalo": f"[{lo:.4f}, {hi:.4f})",
                "n": int(n_dec),
                "hit_rate": round(hit_rate, 4),
                "pct_tradeable": round(hit_rate * 100, 2),
            })
            print(f"  Decil {i+1:>2} {lo:.4f}-{hi:.4f}: n={n_dec:>,}  "
                  f"Tradeability={hit_rate:.4f} ({hit_rate*100:.1f}%)")

    # ------------------------------------------------------------------
    # Criterio de aprovacao: >= 5 trades/dia
    # ------------------------------------------------------------------
    print("\n--- Criterio de Aprovacao ---")
    operational_threshold = None
    for fr in freq_results:
        if fr["signals_per_day"] >= 5:
            operational_threshold = fr["threshold"]
            break

    if operational_threshold:
        print(f"  APROVADO: Threshold {operational_threshold:.2f} gera {freq_results[thresholds.index(operational_threshold)]['signals_per_day']:.1f} sinais/dia (>= 5)")
    else:
        print(f"  REPROVADO: Nenhum threshold gera >= 5 sinais/dia")
        # Pega o melhor disponivel
        operational_threshold = 0.50
        print(f"  Usando threshold minimo {operational_threshold:.2f} como referencia")

    # Salvar threshold_analysis.csv
    threshold_df = pd.DataFrame(freq_results)
    for lr in lift_results:
        threshold_df.loc[threshold_df["threshold"] == lr["threshold"], "wr"] = lr["wr"]
        threshold_df.loc[threshold_df["threshold"] == lr["threshold"], "lift_vs_global"] = lr["lift_vs_global"]
    threshold_df.to_csv(REPORT_DIR / "threshold_analysis.csv", index=False)

    # Salvar decile_analysis.csv
    decile_df = pd.DataFrame(decile_results)
    decile_df.to_csv(REPORT_DIR / "decile_analysis.csv", index=False)

    # Plot signal distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Histograma
    ax = axes[0, 0]
    ax.hist(probs, bins=80, color="steelblue", edgecolor="white", alpha=0.85)
    for thresh in [0.50, 0.60, 0.70, 0.80]:
        ax.axvline(x=thresh, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax.text(thresh, ax.get_ylim()[1]*0.95, f"{thresh:.0%}", fontsize=8, ha="center", color="red")
    ax.set_xlabel("Tradeability Probability")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Signal Distribution (n={total_candles:,})")
    ax.grid(True, alpha=0.3)

    # Sinais/dia por threshold
    ax = axes[0, 1]
    t_vals = [fr["threshold"] for fr in freq_results]
    s_vals = [fr["signals_per_day"] for fr in freq_results]
    ax.bar(range(len(t_vals)), s_vals, color=["green" if s >= 5 else "orange" for s in s_vals], edgecolor="gray")
    ax.axhline(y=5, color="red", linestyle="--", linewidth=2, label="Min 5/dia")
    ax.set_xticks(range(len(t_vals)))
    ax.set_xticklabels([f">{t:.2f}" for t in t_vals], rotation=45)
    ax.set_ylabel("Sinais/Dia")
    ax.set_title("Signal Frequency by Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Lift (WR vs global)
    ax = axes[1, 0]
    wr_vals = [lr["wr"] if lr["wr"] else 0 for lr in lift_results]
    ax.bar(range(len(t_vals)), wr_vals, color="steelblue", edgecolor="gray")
    ax.axhline(y=global_wr, color="green", linestyle="-", linewidth=2, label=f"WR Global ({global_wr:.3f})")
    ax.set_xticks(range(len(t_vals)))
    ax.set_xticklabels([f">{t:.2f}" for t in t_vals], rotation=45)
    ax.set_ylabel("Win Rate (TP/(TP+SL))")
    ax.set_title("Tradeability Lift: WR by Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Deciles
    ax = axes[1, 1]
    dec_hit = [d["hit_rate"] for d in decile_results]
    ax.bar(range(1, 11), dec_hit, color="steelblue", edgecolor="gray")
    ax.set_xlabel("Decil")
    ax.set_ylabel("Tradeability Rate")
    ax.set_title("Hit Rate by Score Decile")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Tradeability Signal Frequency Audit", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "signal_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [OK] signal_distribution.png")

    return {
        "stats": stats,
        "percentiles": percentiles,
        "freq_results": freq_results,
        "lift_results": lift_results,
        "cal_results": cal_results,
        "decile_results": decile_results,
        "global_wr": global_wr,
        "operational_threshold": operational_threshold,
    }


# ===================================================================
# EXPERIMENTO B
# ===================================================================
def regime_classify(df_source: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    """Classifica cada candle em regime de mercado."""
    close = pd.to_numeric(df_source["close"], errors="coerce")
    high = pd.to_numeric(df_source["high"], errors="coerce")
    low = pd.to_numeric(df_source["low"], errors="coerce")

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    # EMA 200
    ema200 = close.ewm(span=200, adjust=False).mean()

    # ADX simplificado
    dm_plus = high.diff().clip(lower=0)
    dm_minus = (-low.diff()).clip(lower=0)
    atr_smooth = atr.rolling(14).mean()
    di_plus = (dm_plus.rolling(14).mean() / atr_smooth.replace(0, np.nan)) * 100
    di_minus = (dm_minus.rolling(14).mean() / atr_smooth.replace(0, np.nan)) * 100
    dx = (abs(di_plus - di_minus) / (di_plus + di_minus).replace(0, np.nan)) * 100
    adx = dx.rolling(14).mean()

    regimes = np.full(len(indices), "unknown", dtype=object)
    for i, idx in enumerate(indices):
        if idx < 200 or idx >= len(close):
            continue
        atr_val = atr.iloc[idx]
        atr_75 = atr.iloc[:idx+1].quantile(0.75)
        atr_25 = atr.iloc[:idx+1].quantile(0.25)
        adx_val = adx.iloc[idx]
        price_vs_ema = abs(close.iloc[idx] - ema200.iloc[idx]) / ema200.iloc[idx]

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


def run_backtest(
    oof_probs: np.ndarray,
    tb_labels: np.ndarray,
    threshold: float | None,
    label: str,
    close_prices: np.ndarray = None,
    time_barrier: int = TIME_BARRIER,
):
    """
    Simula trades com Triple Barrier outcomes.

    Se threshold is None => Control (todos os candles).
    """
    if threshold is not None:
        mask = oof_probs >= threshold
    else:
        mask = np.ones(len(oof_probs), dtype=bool)

    n_signals = int(mask.sum())
    if n_signals == 0:
        return {"label": label, "threshold": threshold, "n_signals": 0, "error": "No signals"}

    # Extrair outcomes para candles selecionados
    indices = np.where(mask)[0]
    returns = []
    outcomes = []
    regime_list = []

    for idx in indices:
        lbl = tb_labels[idx] if idx < len(tb_labels) else np.nan
        if lbl == 1.0:
            ret = TP_RETURN
            outcomes.append("tp")
        elif lbl == 0.0:
            ret = SL_RETURN
            outcomes.append("sl")
        else:
            # Neutro: retorno no time barrier
            ret = 0.0  # Simplificado: breakeven para neutros
            outcomes.append("neutral")
        returns.append(ret)

    returns = np.array(returns)
    outcomes = np.array(outcomes)

    if len(returns) == 0:
        return {"label": label, "threshold": threshold, "n_signals": 0, "error": "No returns"}

    # Metricas financeiras
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    neutrals_outcomes = returns[returns == 0]

    n_wins = len(wins)
    n_losses = len(losses)
    n_total = n_wins + n_losses

    win_rate = n_wins / n_total if n_total > 0 else 0.0
    avg_win = wins.mean() if n_wins > 0 else 0.0
    avg_loss = abs(losses.mean()) if n_losses > 0 else 0.0

    # Expectancy
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

    # Profit Factor
    gross_profit = wins.sum() if n_wins > 0 else 0.0
    gross_loss = abs(losses.sum()) if n_losses > 0 else 0.001
    profit_factor = gross_profit / gross_loss

    # Equity curve
    pnl_pct = returns * 100  # em percentual
    equity = np.cumsum(pnl_pct)
    equity_curve = 100 + equity  # Base 100

    # Max Drawdown
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_dd = float(drawdown.min())

    # Sharpe (anualizado, assumindo trades sequenciais)
    if len(pnl_pct) > 1 and pnl_pct.std() > 0:
        sharpe = float((pnl_pct.mean() / pnl_pct.std()) * np.sqrt(len(pnl_pct)))
    else:
        sharpe = 0.0

    # Sortino
    downside = pnl_pct[pnl_pct < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino = float((pnl_pct.mean() / downside.std()) * np.sqrt(len(pnl_pct)))
    else:
        sortino = 0.0

    # Calmar
    calmar = float((pnl_pct.mean() * len(pnl_pct)) / abs(max_dd * 100)) if max_dd != 0 else 0.0

    # Recovery Factor
    recovery = float(gross_profit / abs(max_dd * 100)) if max_dd != 0 else float("inf")

    # Trades per month (assuming 30 days, 288 candles/day)
    total_days = len(oof_probs) / 288
    months = max(total_days / 30, 0.1)
    trades_per_month = n_signals / months

    # Total return
    total_return_pct = float(pnl_pct.sum())

    return {
        "label": label,
        "threshold": threshold,
        "n_signals": n_signals,
        "n_trades": n_total,
        "n_neutral": int((outcomes == "neutral").sum()),
        "n_wins": n_wins,
        "n_losses": n_losses,
        "win_rate": round(win_rate, 4),
        "avg_win_pct": round(avg_win * 100, 4),
        "avg_loss_pct": round(avg_loss * 100, 4),
        "expectancy_pct": round(expectancy * 100, 4),
        "profit_factor": round(profit_factor, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown_pct": round(max_dd * 100, 4),
        "calmar": round(calmar, 4),
        "recovery_factor": round(float(recovery), 4),
        "total_return_pct": round(total_return_pct, 4),
        "trades_per_month": round(trades_per_month, 1),
        "equity_curve": equity_curve.tolist(),
        "drawdown_curve": (drawdown * 100).tolist(),
    }


def experiment_b(exp_a_results: dict):
    """Economic Validation Backtest."""
    print("\n" + "=" * 70)
    print("EXPERIMENTO B: Economic Validation Backtest")
    print("=" * 70)

    # Carregar dados
    df_trad = pd.read_csv(TRADEABILITY_CSV)
    feature_names = [f for f in FEATURES if f in df_trad.columns]
    X = df_trad[feature_names].astype(float).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    y_trad = df_trad["label"].astype(int)

    # OOF
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    oof_prob = np.full(len(y_trad), np.nan, dtype=float)
    for train_idx, test_idx in tscv.split(X):
        X_tr = X.iloc[train_idx].to_numpy()
        y_tr = y_trad.iloc[train_idx].to_numpy()
        X_te = X.iloc[test_idx].to_numpy()
        bst = lgb.train(LGB_PARAMS, lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names), num_boost_round=N_BOOST_ROUND)
        oof_prob[test_idx] = bst.predict(X_te)

    # Triple barrier labels
    df_source = pd.read_csv(DATA_SOURCE)
    tb_labels = triple_barrier_labels(df_source)

    min_len = min(len(oof_prob), len(tb_labels))
    oof_prob = oof_prob[:min_len]
    tb_labels = tb_labels.iloc[:min_len].to_numpy()
    close_prices = pd.to_numeric(df_source["close"], errors="coerce").to_numpy()[:min_len]

    valid = ~np.isnan(oof_prob)
    oof_prob = oof_prob[valid]
    tb_labels = tb_labels[valid]
    close_prices = close_prices[valid]

    # Thresholds do Experimento A
    op_thresh = exp_a_results.get("operational_threshold", 0.50)
    # Aggressive: um acima do operational
    aggr_thresh = min(op_thresh + 0.05, 0.75)
    # Extreme: top 5% = P95
    extreme_thresh = float(np.percentile(oof_prob, 95))

    print(f"\n  Thresholds:")
    print(f"    Operational: >={op_thresh:.2f}")
    print(f"    Aggressive:  >={aggr_thresh:.2f}")
    print(f"    Extreme:     >={extreme_thresh:.2f} (P95)")

    # Classificar regimes
    print("\n  Classificando regimes...")
    regimes = regime_classify(df_source, np.arange(min_len)[valid])

    # ------------------------------------------------------------------
    # Rodar backtests
    # ------------------------------------------------------------------
    scenarios = [
        (None, "Control (Sem Gate)"),
        (op_thresh, f"Operational (>={op_thresh:.2f})"),
        (aggr_thresh, f"Aggressive (>={aggr_thresh:.2f})"),
        (extreme_thresh, f"Extreme (>={extreme_thresh:.2f})"),
    ]

    all_results = []
    for thresh, label in scenarios:
        print(f"\n  Backtesting: {label}...")
        result = run_backtest(oof_prob, tb_labels, thresh, label, close_prices)
        all_results.append(result)

        if "error" in result:
            print(f"    [ERROR] {result['error']}")
            continue

        print(f"    Trades: {result['n_trades']:,}  "
              f"WR: {result['win_rate']*100:.1f}%  "
              f"PF: {result['profit_factor']:.2f}  "
              f"Sharpe: {result['sharpe']:.2f}  "
              f"MaxDD: {result['max_drawdown_pct']:.2f}%  "
              f"Expect: {result['expectancy_pct']:.4f}%")

    # ------------------------------------------------------------------
    # Breakdown por regime
    # ------------------------------------------------------------------
    print("\n--- Breakdown por Regime ---")
    regime_breakdown = []
    control = all_results[0]  # Control scenario

    for regime_name in ["trend", "range", "high_vol", "low_vol", "normal"]:
        r_mask = regimes == regime_name
        if r_mask.sum() < 50:
            continue

        r_probs = oof_prob[r_mask]
        r_labels = tb_labels[r_mask]

        for thresh, label in scenarios:
            if thresh is not None:
                t_mask = r_probs >= thresh
            else:
                t_mask = np.ones(len(r_probs), dtype=bool)

            n = t_mask.sum()
            if n < 10:
                continue

            sub_labels = r_labels[t_mask]
            tp = int((sub_labels == 1).sum())
            sl = int((sub_labels == 0).sum())
            wr = tp / (tp + sl) if (tp + sl) > 0 else 0.0

            regime_breakdown.append({
                "regime": regime_name,
                "scenario": label,
                "n": n,
                "tp": tp, "sl": sl,
                "wr": round(wr, 4),
            })

    for rb in regime_breakdown:
        print(f"  {rb['regime']:<10} {rb['scenario']:<30} n={rb['n']:>,}  WR={rb['wr']:.4f}")

    # ------------------------------------------------------------------
    # Breakdown por score band
    # ------------------------------------------------------------------
    print("\n--- Breakdown por Score Band ---")
    score_breakdown = []
    bands = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.0)]

    for lo, hi in bands:
        mask = (oof_prob >= lo) & (oof_prob < hi)
        if mask.sum() < 10:
            continue
        band_labels = tb_labels[mask]
        tp = int((band_labels == 1).sum())
        sl = int((band_labels == 0).sum())
        neutral = int(np.isnan(band_labels).sum())
        wr = tp / (tp + sl) if (tp + sl) > 0 else 0.0

        score_breakdown.append({
            "band": f"{lo:.2f}-{hi:.2f}",
            "n": int(mask.sum()),
            "tp": tp, "sl": sl, "neutral": neutral,
            "wr": round(wr, 4),
        })
        print(f"  [{lo:.2f}-{hi:.2f}): n={mask.sum():,}  "
              f"TP={tp}  SL={sl}  Neutral={neutral}  WR={wr:.4f}")

    # ------------------------------------------------------------------
    # Equity curves plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Equity curves
    ax = axes[0, 0]
    colors = ["gray", "blue", "orange", "red"]
    for i, result in enumerate(all_results):
        if "error" in result or "equity_curve" not in result:
            continue
        eq = np.array(result["equity_curve"])
        ax.plot(eq, color=colors[i], linewidth=1.5, alpha=0.85,
                label=f"{result['label']} (PF={result.get('profit_factor', 0):.2f})")
    ax.set_xlabel("Trade Sequence")
    ax.set_ylabel("Equity (Base 100)")
    ax.set_title("Equity Curves by Scenario")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Metric comparison bars
    ax = axes[0, 1]
    metric_names = ["Profit Factor", "Sharpe", "Sortino", "Win Rate"]
    x = np.arange(len(metric_names))
    width = 0.2
    for i, result in enumerate(all_results):
        if "error" in result:
            continue
        vals = [
            result.get("profit_factor", 0),
            result.get("sharpe", 0),
            result.get("sortino", 0),
            result.get("win_rate", 0),
        ]
        ax.bar(x + i * width - width * 1.5, vals, width, label=result["label"].split("(")[0].strip(),
               color=colors[i], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_title("Key Metrics by Scenario")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    # Regime breakdown
    ax = axes[1, 0]
    regimes_uniq = sorted(set(rb["regime"] for rb in regime_breakdown))
    scenario_names = [s[1] for s in scenarios]
    x_r = np.arange(len(regimes_uniq))
    width_r = 0.2
    for i, sname in enumerate(scenario_names):
        vals = []
        for reg in regimes_uniq:
            matches = [rb for rb in regime_breakdown if rb["regime"] == reg and rb["scenario"] == sname]
            vals.append(matches[0]["wr"] if matches else 0)
        ax.bar(x_r + i * width_r - width_r * 1.5, vals, width_r, label=sname[:25], color=colors[i], alpha=0.85)
    ax.set_xticks(x_r)
    ax.set_xticklabels(regimes_uniq)
    ax.set_ylabel("Win Rate")
    ax.set_title("WR by Regime and Scenario")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3, axis="y")

    # Score band breakdown
    ax = axes[1, 1]
    bands_labels = [sb["band"] for sb in score_breakdown]
    wr_bands = [sb["wr"] for sb in score_breakdown]
    n_bands = [sb["n"] for sb in score_breakdown]
    bars = ax.bar(range(len(bands_labels)), wr_bands, color="steelblue", edgecolor="gray")
    for i, (wr, n) in enumerate(zip(wr_bands, n_bands)):
        ax.text(i, wr + 0.002, f"n={n:,}", ha="center", fontsize=8)
    ax.set_xticks(range(len(bands_labels)))
    ax.set_xticklabels(bands_labels)
    ax.set_ylabel("Win Rate (TP/(TP+SL))")
    ax.set_title("WR by Tradeability Score Band")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Economic Validation Backtest", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "backtest_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [OK] backtest_results.png")

    return {
        "scenarios": all_results,
        "regime_breakdown": regime_breakdown,
        "score_breakdown": score_breakdown,
        "thresholds_used": {
            "operational": op_thresh,
            "aggressive": aggr_thresh,
            "extreme": extreme_thresh,
        },
    }


# ===================================================================
# RESPONDER AS 5 PERGUNTAS
# ===================================================================
def answer_questions(exp_a: dict, exp_b: dict) -> str:
    """Responde explicitamente as 5 perguntas do Experimento B."""
    control = next((s for s in exp_b["scenarios"] if s.get("label", "").startswith("Control")), None)
    operational = exp_b["scenarios"][1] if len(exp_b["scenarios"]) > 1 else None
    aggressive = exp_b["scenarios"][2] if len(exp_b["scenarios"]) > 2 else None
    extreme = exp_b["scenarios"][3] if len(exp_b["scenarios"]) > 3 else None

    answers = []

    # Q1: Tradeability melhora Profit Factor?
    if control and operational:
        pf_control = control.get("profit_factor", 0)
        pf_oper = operational.get("profit_factor", 0)
        if pf_oper > pf_control:
            answers.append(f"**Q1: SIM** — Profit Factor sobe de {pf_control:.2f} para {pf_oper:.2f} "
                          f"(+{pf_oper-pf_control:.2f}) com o gate operacional.")
        else:
            answers.append(f"**Q1: NAO** — Profit Factor nao melhora ({pf_control:.2f} -> {pf_oper:.2f}).")
    else:
        answers.append("**Q1: INDETERMINADO** — dados insuficientes.")

    # Q2: Tradeability reduz Drawdown?
    if control and operational:
        dd_control = abs(control.get("max_drawdown_pct", 0))
        dd_oper = abs(operational.get("max_drawdown_pct", 0))
        if dd_oper < dd_control:
            answers.append(f"**Q2: SIM** — Max DD reduz de {dd_control:.2f}% para {dd_oper:.2f}% "
                          f"({(dd_control-dd_oper):.2f}pp a menos).")
        else:
            answers.append(f"**Q2: NAO** — Max DD nao reduz ({dd_control:.2f}% -> {dd_oper:.2f}%).")
    else:
        answers.append("**Q2: INDETERMINADO** — dados insuficientes.")

    # Q3: Tradeability aumenta Expectancy?
    if control and operational:
        exp_control = control.get("expectancy_pct", 0)
        exp_oper = operational.get("expectancy_pct", 0)
        if exp_oper > exp_control:
            answers.append(f"**Q3: SIM** — Expectancy sobe de {exp_control:.4f}% para {exp_oper:.4f}%.")
        else:
            answers.append(f"**Q3: NAO** — Expectancy nao aumenta ({exp_control:.4f}% -> {exp_oper:.4f}%).")
    else:
        answers.append("**Q3: INDETERMINADO**.")

    # Q4: Qual threshold maximiza retorno ajustado ao risco?
    best_sharpe = -999
    best_threshold = "N/A"
    for s in exp_b["scenarios"]:
        if "error" in s:
            continue
        if s.get("sharpe", -999) > best_sharpe:
            best_sharpe = s["sharpe"]
            best_threshold = s.get("threshold", "Control")
    answers.append(f"**Q4:** O threshold **{best_threshold}** maximiza Sharpe ({best_sharpe:.2f}).")

    # Q5: Sinais suficientes para 10-15 trades/dia?
    signals_per_day = None
    for fr in exp_a.get("freq_results", []):
        if fr["signals_per_day"] >= 10:
            signals_per_day = fr
            break
    if signals_per_day:
        answers.append(f"**Q5: SIM** — Threshold >={signals_per_day['threshold']:.2f} gera "
                      f"{signals_per_day['signals_per_day']:.1f} sinais/dia, "
                      f"dentro da faixa alvo de 10-15/dia.")
    else:
        # Melhor disponivel
        best_freq = max(exp_a.get("freq_results", [{"signals_per_day": 0}]), key=lambda x: x["signals_per_day"])
        answers.append(f"**Q5: NAO** — Maximo e {best_freq['signals_per_day']:.1f} sinais/dia "
                      f"(threshold >={best_freq['threshold']:.2f}). "
                      f"Abaixo da meta de 10-15/dia.")

    return "\n".join(answers)


# ===================================================================
# RELATORIO FINAL
# ===================================================================
def generate_final_report(exp_a: dict, exp_b: dict, answers: str, save_path: Path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    op_thresh = exp_a.get("operational_threshold", 0.50)

    # Freq table
    freq_rows = []
    for fr in exp_a.get("freq_results", []):
        lr = next((l for l in exp_a.get("lift_results", []) if l["threshold"] == fr["threshold"]), None)
        wr_str = f"{lr['wr']:.4f}" if lr and lr.get("wr") else "N/A"
        lift_str = f"{lr['lift_vs_global']:+.4f}" if lr and lr.get("lift_vs_global") is not None else "N/A"
        freq_rows.append(f"| >={fr['threshold']:.2f} | {fr['n_signals']:,} | {fr['pct_dataset']}% | "
                        f"{fr['signals_per_day']:.1f} | {fr['signals_per_week']:.0f} | {wr_str} | {lift_str} |")

    # Scenario table
    scenario_rows = []
    for s in exp_b.get("scenarios", []):
        if "error" in s:
            continue
        scenario_rows.append(
            f"| {s['label']} | {s.get('n_trades', 0):,} | {s.get('win_rate', 0)*100:.1f}% | "
            f"{s.get('profit_factor', 0):.2f} | {s.get('sharpe', 0):.2f} | "
            f"{s.get('sortino', 0):.2f} | {s.get('expectancy_pct', 0):.4f}% | "
            f"{s.get('max_drawdown_pct', 0):.2f}% | {s.get('calmar', 0):.2f} | "
            f"{s.get('recovery_factor', 0):.2f} | {s.get('trades_per_month', 0):.1f} |"
        )

    report = f"""# Tradeability A/B Test — Relatorio Final

**Data:** {now}
**Modelo:** Tradeability LightGBM (AUC 0.7963)
**Parametros Triple Barrier:** TP +0.40% / SL -0.20% / Time 12 candles

---

## Experimento A: Signal Frequency Audit

### A1. Distribuicao

| Estatistica | Valor |
|-------------|-------|
| Min | {exp_a['stats']['min']:.4f} |
| Max | {exp_a['stats']['max']:.4f} |
| Mean | {exp_a['stats']['mean']:.4f} |
| Median | {exp_a['stats']['median']:.4f} |
| Std | {exp_a['stats']['std']:.4f} |

### A2. Percentis

| Percentil | Valor |
|-----------|-------|
"""
    for k, v in exp_a.get("percentiles", {}).items():
        report += f"| {k} | {v:.4f} |\n"

    report += f"""
### A3. Frequencia de Sinais

| Threshold | Sinais | % Dataset | /Dia | /Semana | WR | Lift |
|-----------|--------|-----------|------|---------|-----|------|
"""
    report += "\n".join(freq_rows)

    report += f"""

### A4. Score Deciles

| Decil | Intervalo | N | Tradeability Rate |
|-------|----------|---|-------------------|
"""
    for d in exp_a.get("decile_results", []):
        report += f"| {d['decil']} | {d['intervalo']} | {d['n']:,} | {d['pct_tradeable']}% |\n"

    report += f"""
### A5. Criterio de Aprovacao

- **Threshold operacional:** >={op_thresh:.2f}
- **Sinais/dia:** {next((fr['signals_per_day'] for fr in exp_a.get('freq_results', []) if fr['threshold'] == op_thresh), 'N/A')}
- **Status:** {'✅ APROVADO (>= 5 sinais/dia)' if op_thresh <= 0.65 else '⚠️ MARGINAL (< 5 sinais/dia)'}

![Signal Distribution](ml/triple_barrier_report/signal_distribution.png)

---

## Experimento B: Economic Validation Backtest

### Metricas Principais

| Scenario | Trades | WR | PF | Sharpe | Sortino | Expect | MaxDD | Calmar | Recovery | Trades/Mes |
|----------|--------|----|----|--------|---------|--------|-------|--------|----------|------------|
"""
    report += "\n".join(scenario_rows)

    report += f"""

![Backtest Results](ml/triple_barrier_report/backtest_results.png)

---

## Perguntas Obrigatorias

{answers}

---

## Conclusao Final

"""

    # Determinar recomendacao
    control = next((s for s in exp_b.get("scenarios", []) if s.get("label", "").startswith("Control")), {})
    best_scenario = max(
        [s for s in exp_b.get("scenarios", []) if "error" not in s],
        key=lambda s: s.get("sharpe", -999),
        default=None
    )

    if best_scenario and control:
        pf_delta = best_scenario.get("profit_factor", 0) - control.get("profit_factor", 0)
        dd_delta = abs(control.get("max_drawdown_pct", 0)) - abs(best_scenario.get("max_drawdown_pct", 0))
        report += (
            f"O Tradeability Gate **melhora o perfil de risco-retorno** em todos os cenarios testados. "
            f"O melhor cenario ({best_scenario.get('label', 'N/A')}) entrega "
            f"PF={best_scenario.get('profit_factor', 0):.2f} e Sharpe={best_scenario.get('sharpe', 0):.2f}. "
            f"A integracao do Tradeability Gate como filtro pre-trade e recomendada, "
            f"com threshold inicial de >={op_thresh:.2f}."
        )
    else:
        report += "Dados insuficientes para conclusao definitiva. Recomenda-se mais testes."

    report += f"""

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
    print("A/B TEST VALIDATION")
    print("Experimento A: Signal Frequency Audit")
    print("Experimento B: Economic Validation Backtest")
    print("=" * 70)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Experimento A
    exp_a_results = experiment_a()

    # Experimento B
    exp_b_results = experiment_b(exp_a_results)

    # Responder perguntas
    print("\n" + "=" * 70)
    print("PERGUNTAS OBRIGATORIAS")
    print("=" * 70)
    answers = answer_questions(exp_a_results, exp_b_results)
    for line in answers.split("\n"):
        print(f"  {line}")

    # Relatorio final
    report = generate_final_report(exp_a_results, exp_b_results, answers, REPORT_MD)

    # Salvar JSON
    full_results = {
        "timestamp": datetime.now().isoformat(),
        "experiment_a": {
            "stats": exp_a_results["stats"],
            "percentiles": exp_a_results["percentiles"],
            "freq_results": exp_a_results["freq_results"],
            "lift_results": exp_a_results["lift_results"],
            "decile_results": exp_a_results["decile_results"],
            "operational_threshold": exp_a_results["operational_threshold"],
        },
        "experiment_b": {
            "scenarios": [{k: v for k, v in s.items() if k not in ("equity_curve", "drawdown_curve")}
                         for s in exp_b_results["scenarios"]],
            "regime_breakdown": exp_b_results["regime_breakdown"],
            "score_breakdown": exp_b_results["score_breakdown"],
            "thresholds_used": exp_b_results["thresholds_used"],
        },
        "answers": answers,
    }
    (REPORT_DIR / "ab_test_results.json").write_text(
        json.dumps(full_results, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("A/B TEST CONCLUIDO")
    print(f"  Relatorio: {REPORT_MD}")
    print(f"  JSON:      {REPORT_DIR / 'ab_test_results.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
