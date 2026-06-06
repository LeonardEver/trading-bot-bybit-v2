"""
EXPERIMENTO 001 — Tradeability Model

Hipotese: o modelo tem mais capacidade de prever "oportunidades operaveis"
(se o candle vai atingir QUALQUER barreira) do que direcao (TP vs SL).

Target binario:
  Classe 0 = Neutro (nenhuma barreira atingida em 12 candles)
  Classe 1 = Tradeable (TP ou SL atingido)

Features, hiperparametros, e TimeSeriesSplit identicos ao Triple Barrier v2.

Comparacao: Tradeability vs Triple Barrier Direction.
Criterio de aprovacao: AUC >= 0.70.
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
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit

from ml.ml_data_pipeline_v2 import (
    triple_barrier_labels,
    UPPER_BARRIER,
    LOWER_BARRIER,
    TIME_BARRIER,
    to_timestamp_ms,
)
from ml.features import prepare_features, apply_strict_feature_lag
from ml.config import FEATURES

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_SOURCE = ROOT / "dataset.csv"
TRADEABILITY_CSV = ROOT / "ml" / "dataset_tradeability.csv"
MODEL_OUT = ROOT / "ml" / "model_tradeability.pkl"
REPORT_DIR = ROOT / "ml" / "triple_barrier_report"
REPORT_MD = ROOT / "tradeability_model_report.md"

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "seed": 42,
}
N_BOOST_ROUND = 300
N_SPLITS = 5

# Caminho do modelo Triple Barrier Direction para comparacao
TB_MODEL_METRICS = REPORT_DIR / "metrics.json"


# ===================================================================
# 1. Geracao do Dataset Tradeability
# ===================================================================
def build_tradeability_dataset(source_csv: Path) -> pd.DataFrame:
    """
    Cria dataset com target binario:
      0 = Neutro (nenhuma barreira)
      1 = Tradeable (TP ou SL)
    """
    print("=" * 60)
    print("Gerando dataset Tradeability")
    print("=" * 60)

    df = pd.read_csv(source_csv)
    raw_labels = triple_barrier_labels(df)

    # Tradeability: 1 se qualquer barreira foi atingida, 0 se neutro
    tradeability_label = raw_labels.notna().astype(int)

    tradeable = int((tradeability_label == 1).sum())
    neutral = int((tradeability_label == 0).sum())
    print(f"  Tradeable (1): {tradeable:,} ({tradeable/len(df)*100:.1f}%)")
    print(f"  Neutro (0):    {neutral:,} ({neutral/len(df)*100:.1f}%)")
    print(f"  Total:         {len(df):,}")

    # Features
    df_feat = prepare_features(df.copy())
    df_feat["label"] = tradeability_label.values

    # Lag estrito
    df_feat = apply_strict_feature_lag(df_feat, FEATURES, periods=1)

    # Garantir features existentes
    existing_features = [f for f in FEATURES if f in df_feat.columns]

    # Remover NaNs
    df_clean = df_feat.dropna(subset=existing_features + ["label"]).reset_index(drop=True)

    tradeable_clean = int((df_clean["label"] == 1).sum())
    neutral_clean = int((df_clean["label"] == 0).sum())
    print(f"  Apos limpeza: {len(df_clean):,} linhas")
    print(f"    Tradeable: {tradeable_clean:,} ({tradeable_clean/len(df_clean)*100:.1f}%)")
    print(f"    Neutro:    {neutral_clean:,} ({neutral_clean/len(df_clean)*100:.1f}%)")
    print()

    # Salvar
    df_clean.to_csv(TRADEABILITY_CSV, index=False)
    print(f"  [OK] Dataset -> {TRADEABILITY_CSV}\n")
    return df_clean


# ===================================================================
# 2. Walk-Forward Evaluation
# ===================================================================
def walk_forward_evaluate(X: np.ndarray, y: np.ndarray, feature_names: list[str], label: str) -> dict:
    """TimeSeriesSplit walk-forward com LightGBM."""
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    oof_pred = np.full(len(y), np.nan, dtype=float)
    fold_metrics = []
    best_auc = -1.0
    best_model = None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        bst = lgb.train(LGB_PARAMS, train_data, num_boost_round=N_BOOST_ROUND)
        y_pred = bst.predict(X_test)
        oof_pred[test_idx] = y_pred

        y_pred_bin = (y_pred >= 0.5).astype(int)
        auc_fold = roc_auc_score(y_test, y_pred)
        pr_auc_fold = average_precision_score(y_test, y_pred)
        acc_fold = accuracy_score(y_test, y_pred_bin)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        ll_fold = log_loss(y_test, y_pred_clipped)
        brier_fold = brier_score_loss(y_test, y_pred_clipped)

        fold_metrics.append({
            "fold": int(fold),
            "auc": float(auc_fold),
            "pr_auc": float(pr_auc_fold),
            "accuracy": float(acc_fold),
            "log_loss": float(ll_fold),
            "brier_score": float(brier_fold),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "label_dist_test": {"0": int((y_test == 0).sum()), "1": int((y_test == 1).sum())},
        })

        print(f"  {label} Fold {fold}: AUC={auc_fold:.4f}  PR_AUC={pr_auc_fold:.4f}  "
              f"ACC={acc_fold:.4f}  LogLoss={ll_fold:.4f}  Brier={brier_fold:.4f}")

        if auc_fold > best_auc:
            best_auc = auc_fold
            best_model = bst

    valid = ~np.isnan(oof_pred)
    y_valid = y[valid]
    p_valid = oof_pred[valid]
    p_valid_clipped = np.clip(p_valid, 1e-15, 1 - 1e-15)

    overall = {
        "auc": float(roc_auc_score(y_valid, p_valid)),
        "pr_auc": float(average_precision_score(y_valid, p_valid)),
        "accuracy": float(accuracy_score(y_valid, (p_valid >= 0.5).astype(int))),
        "log_loss": float(log_loss(y_valid, p_valid_clipped)),
        "brier_score": float(brier_score_loss(y_valid, p_valid_clipped)),
        "n_valid": int(len(y_valid)),
    }

    print(f"  {label} Overall: AUC={overall['auc']:.4f}  PR_AUC={overall['pr_auc']:.4f}  "
          f"ACC={overall['accuracy']:.4f}  LogLoss={overall['log_loss']:.4f}  "
          f"Brier={overall['brier_score']:.4f}\n")

    return {
        "label": label,
        "fold_metrics": fold_metrics,
        "overall": overall,
        "oof_predictions": p_valid,
        "y_true": y_valid,
        "model": best_model,
    }


# ===================================================================
# 3. SHAP
# ===================================================================
def compute_shap(model, X_sample: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    try:
        contributions = model.predict(X_sample, pred_contrib=True)
        shap_values = contributions[:, :-1]
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        return pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False)
    except Exception as e:
        print(f"  [WARN] SHAP: {e}")
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])


# ===================================================================
# 4. Plots comparativos
# ===================================================================
def plot_comparative_roc(results_trad: dict, results_tb: dict, save_path: Path):
    """ROC curves: Tradeability vs Triple Barrier Direction."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, res, title in [
        (axes[0], results_trad, "Tradeability (Neutro vs Tradeable)"),
        (axes[1], results_tb, "Triple Barrier Direction (TP vs SL)"),
    ]:
        y_true = res["y_true"]
        y_pred = res["oof_predictions"]
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_val = res["overall"]["auc"]

        ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC={auc_val:.4f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.fill_between(fpr, tpr, alpha=0.1, color="blue")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("ROC Curves — Tradeability vs Triple Barrier Direction",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] ROC -> {save_path}")


def plot_pr_curves(results_trad: dict, results_tb: dict, save_path: Path):
    """Precision-Recall curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, res, title in [
        (axes[0], results_trad, "Tradeability"),
        (axes[1], results_tb, "Triple Barrier Direction"),
    ]:
        y_true = res["y_true"]
        y_pred = res["oof_predictions"]
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = res["overall"]["pr_auc"]
        baseline = y_true.mean()

        ax.plot(recall, precision, "b-", linewidth=2, label=f"PR (AUC={pr_auc:.4f})")
        ax.axhline(y=baseline, color="r", linestyle="--", alpha=0.5,
                   label=f"Baseline ({(baseline*100):.1f}%)")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{title}\nPR AUC: {pr_auc:.4f}")
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Precision-Recall Curves — Tradeability vs Triple Barrier Direction",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] PR Curves -> {save_path}")


def plot_calibration_comparison(results_trad: dict, results_tb: dict, save_path: Path):
    """Calibration curves lado a lado."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, res, title in [
        (axes[0], results_trad, "Tradeability"),
        (axes[1], results_tb, "Triple Barrier Direction"),
    ]:
        y_true = res["y_true"]
        y_pred = res["oof_predictions"]
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10, strategy="uniform")
        brier = res["overall"]["brier_score"]

        ax.plot(prob_pred, prob_true, "b-", linewidth=2, marker="o", markersize=6)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfeita")
        ax.fill_between(prob_pred, prob_true, prob_pred, alpha=0.1, color="blue")
        ax.set_xlabel("Probabilidade Prevista")
        ax.set_ylabel("Frequencia Observada")
        ax.set_title(f"{title}\nBrier: {brier:.4f}")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle("Calibration — Tradeability vs Triple Barrier Direction",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Calibration -> {save_path}")


def plot_shap_top_features(shap_df: pd.DataFrame, save_path: Path, top_n: int = 20):
    """SHAP top features horizontal bar chart."""
    if shap_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    top = shap_df.head(top_n).iloc[::-1]
    colors = plt.cm.RdYlGn(top["mean_abs_shap"] / max(top["mean_abs_shap"].max(), 1e-6))
    ax.barh(range(len(top)), top["mean_abs_shap"], color=colors, edgecolor="gray", alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=9)
    ax.set_xlabel("Mean(|SHAP|)")
    ax.set_title(f"Tradeability Model — Top {top_n} SHAP Features", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="x")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] SHAP -> {save_path}")


def plot_fold_metrics(results_trad: dict, results_tb: dict, save_path: Path):
    """Metricas por fold: Tradeability vs TB Direction."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    folds_trad = [m["fold"] for m in results_trad["fold_metrics"]]
    folds_tb = [m["fold"] for m in results_tb["fold_metrics"]]

    metrics = ["auc", "pr_auc", "accuracy", "log_loss", "brier_score"]
    titles = ["ROC AUC", "PR AUC", "Accuracy", "LogLoss", "Brier Score"]
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

    for (r, c), metric, title in zip(positions, metrics, titles):
        ax = axes[r, c]
        vals_trad = [m[metric] for m in results_trad["fold_metrics"]]
        vals_tb = [m[metric] for m in results_tb["fold_metrics"]]
        ax.plot(folds_trad, vals_trad, "go-", linewidth=2, markersize=8, label="Tradeability")
        ax.plot(folds_tb, vals_tb, "bo-", linewidth=2, markersize=8, label="TB Direction")
        ax.set_xlabel("Fold")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Esconder subplot extra
    axes[1, 2].set_visible(False)

    fig.suptitle("Metricas por Fold — Tradeability vs Triple Barrier Direction",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Fold Metrics -> {save_path}")


# ===================================================================
# 5. Relatorio
# ===================================================================
def generate_report(
    results_trad: dict,
    results_tb: dict,
    shap_trad: pd.DataFrame,
    passed: bool,
    save_path: Path,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ot = results_trad["overall"]
    ob = results_tb["overall"]

    delta_auc = ot["auc"] - ob["auc"]
    delta_pr = ot["pr_auc"] - ob["pr_auc"]
    delta_brier = ot["brier_score"] - ob["brier_score"]

    auc_pass = ot["auc"] >= 0.70

    # Top-10 SHAP
    shap_rows = []
    if not shap_trad.empty:
        for i, (_, row) in enumerate(shap_trad.head(20).iterrows()):
            shap_rows.append(f"| {i+1} | `{row['feature']}` | {row['mean_abs_shap']:.6f} |")

    report = f"""# Tradeability Model — Relatorio do Experimento 001

**Data:** {now}
**Hipotese:** O modelo tem mais capacidade de prever "oportunidades operaveis"
(tradeability) do que direcao (TP vs SL).

**Target:**
- Classe 0 = Neutro (nenhuma barreira atingida em {TIME_BARRIER} candles)
- Classe 1 = Tradeable (TP ou SL atingido)

**Parametros:** LightGBM identico ao Triple Barrier v2 | TimeSeriesSplit 5 folds

---

## 1. Resultado Principal

| Metrica | Tradeability | TB Direction | Delta |
|---------|-------------|--------------|-------|
| **ROC AUC** | **{ot['auc']:.4f}** | {ob['auc']:.4f} | {delta_auc:+.4f} |
| **PR AUC** | **{ot['pr_auc']:.4f}** | {ob['pr_auc']:.4f} | {delta_pr:+.4f} |
| **Accuracy** | {ot['accuracy']:.4f} | {ob['accuracy']:.4f} | — |
| **LogLoss** | {ot['log_loss']:.4f} | {ob['log_loss']:.4f} | — |
| **Brier Score** | {ot['brier_score']:.4f} | {ob['brier_score']:.4f} | {delta_brier:+.4f} |

### Criterio de Aprovacao: AUC >= 0.70

| Criterio | Threshold | Observado | Status |
|----------|-----------|-----------|--------|
| ROC AUC | >= 0.70 | {ot['auc']:.4f} | {"✅ APROVADO" if auc_pass else "❌ REPROVADO"} |

---

## 2. Metricas por Fold

| Fold | AUC | PR AUC | Accuracy | LogLoss | Brier |
|------|-----|--------|----------|---------|-------|
"""
    for m in results_trad["fold_metrics"]:
        report += f"| {m['fold']} | {m['auc']:.4f} | {m['pr_auc']:.4f} | {m['accuracy']:.4f} | {m['log_loss']:.4f} | {m['brier_score']:.4f} |\n"

    report += f"""
- **AUC Media:** {np.mean([m['auc'] for m in results_trad['fold_metrics']]):.4f} +/- {np.std([m['auc'] for m in results_trad['fold_metrics']]):.4f}
- **PR AUC Media:** {np.mean([m['pr_auc'] for m in results_trad['fold_metrics']]):.4f} +/- {np.std([m['pr_auc'] for m in results_trad['fold_metrics']]):.4f}

---

## 3. SHAP Feature Importance (Tradeability)

| Rank | Feature | Mean(|SHAP|) |
|------|---------|---------------|
"""
    report += "\n".join(shap_rows) if shap_rows else "| — | SHAP nao disponivel | — |"

    report += f"""

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

"""

    if delta_auc > 0.05:
        report += (
            "**SIM, fortemente.** O modelo de tradeability apresenta AUC "
            f"significativamente superior ({ot['auc']:.4f} vs {ob['auc']:.4f}, "
            f"delta = {delta_auc:+.4f}). Isso confirma que o modelo tem mais capacidade "
            "de identificar volatilidade/oportunidade do que direcao."
        )
    elif delta_auc > 0.01:
        report += (
            "**SIM, moderadamente.** O modelo de tradeability apresenta AUC superior "
            f"({ot['auc']:.4f} vs {ob['auc']:.4f}), mas a diferenca e modesta. "
            "Ha evidencias de que o modelo captura melhor oportunidades do que direcao, "
            "porem o ganho nao e dramatico."
        )
    else:
        report += (
            "**NAO conclusivamente.** A diferenca de AUC e pequena ou inexistente "
            f"({ot['auc']:.4f} vs {ob['auc']:.4f}). Nao ha evidencias fortes de que "
            "o modelo seja substancialmente melhor em prever tradeability do que direcao."
        )

    report += f"""

### 5.2 Implicacoes Praticas

- Um modelo de tradeability com AUC={ot['auc']:.4f} pode ser usado para:
  - **Filtro de entrada:** evitar operar em candles com baixa probabilidade de atingir barreiras
  - **Gestao de risco:** reduzir exposicao em periodos de baixa volatilidade prevista
  - **Sizeamento:** aumentar posicao quando tradeability e alta

### 5.3 Limitacoes

- O target e definido pelos mesmos parametros do Triple Barrier (TP +0.40%, SL -0.20%, 12 candles)
- A classe 1 (tradeable) inclui tanto TP quanto SL — nao distingue oportunidades boas de ruins
- Performance em producao depende da estabilidade do regime de volatilidade

---

## 6. Conclusao

"""

    if auc_pass:
        report += (
            f"**APROVADO** — AUC de {ot['auc']:.4f} atinge o criterio minimo de 0.70. "
            "O modelo de tradeability demonstra capacidade preditiva util e pode ser "
            "integrado ao pipeline como filtro complementar ao modelo de direcao. "
            "Recomenda-se paper trading com o modelo combinado (tradeability gate + "
            "direction model)."
        )
    else:
        report += (
            f"**REPROVADO** — AUC de {ot['auc']:.4f} nao atinge o criterio minimo de 0.70. "
            "Apesar de ser superior ao modelo de direcao, a capacidade preditiva "
            "ainda e insuficiente para uso em producao. Recomenda-se investigar "
            "features adicionais especificas para volatilidade (ex: term structure "
            "de volatilidade, volatility risk premium, etc.)."
        )

    report += f"""

---

*Relatorio gerado automaticamente — {now}*
*Modelo: LightGBM | Features: identicas ao v1 | Target: Tradeability (Neutro vs Tradeable)*
"""

    save_path.write_text(report, encoding="utf-8")
    print(f"\n  [OK] Relatorio -> {save_path}")
    return report


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 70)
    print("EXPERIMENTO 001 — Tradeability Model")
    print("=" * 70)
    print(f"  Hipotese: modelo preve melhor 'oportunidade' do que 'direcao'")
    print(f"  Target: 0=Neutro, 1=Tradeable (TP ou SL)")
    print(f"  Criterio: AUC >= 0.70")
    print("=" * 70)
    print()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Gerar dataset
    # ------------------------------------------------------------------
    print("[1/5] Gerando dataset Tradeability...")
    df_trad = build_tradeability_dataset(DATA_SOURCE)

    feature_names = [f for f in FEATURES if f in df_trad.columns]
    X_trad = df_trad[feature_names].astype(float).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).to_numpy()
    y_trad = df_trad["label"].astype(int).to_numpy()

    # ------------------------------------------------------------------
    # 2. Treinar e avaliar Tradeability
    # ------------------------------------------------------------------
    print("[2/5] Walk-Forward — Tradeability Model...")
    results_trad = walk_forward_evaluate(X_trad, y_trad, feature_names, "Tradeability")

    # Salvar modelo
    if results_trad["model"] is not None:
        joblib.dump(results_trad["model"], MODEL_OUT)
        print(f"  [OK] Modelo -> {MODEL_OUT}")

    # ------------------------------------------------------------------
    # 3. Carregar resultados do TB Direction
    # ------------------------------------------------------------------
    print("[3/5] Carregando resultados do Triple Barrier Direction...")
    if TB_MODEL_METRICS.exists():
        tb_json = json.loads(TB_MODEL_METRICS.read_text(encoding="utf-8"))
        print(f"  TB Direction AUC: {tb_json['triple_barrier']['auc']:.4f}")
    else:
        print("  [WARN] metrics.json nao encontrado — usando valores da sessao anterior")
        tb_json = {
            "triple_barrier": {
                "auc": 0.5396, "accuracy": 0.6949,
                "log_loss": 0.6330, "brier_score": 0.2136,
            }
        }

    # Para comparacao visual, preciso dos OOF preds do TB Direction
    # Vou carregar do dataset_triple_barrier e rodar walk-forward rapido
    print("  Recomputando OOF preds do TB Direction para comparacao...")
    df_tb = pd.read_csv(ROOT / "ml" / "dataset_triple_barrier.csv")
    X_tb = df_tb[feature_names].astype(float).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).to_numpy()
    y_tb = df_tb["label"].astype(int).to_numpy()
    results_tb = walk_forward_evaluate(X_tb, y_tb, feature_names, "TB_Dir")

    # ------------------------------------------------------------------
    # 4. SHAP
    # ------------------------------------------------------------------
    print("[4/5] SHAP Analysis...")
    n_shap = min(2000, X_trad.shape[0])
    shap_trad = compute_shap(results_trad["model"], X_trad[-n_shap:], feature_names)

    # ------------------------------------------------------------------
    # 5. Graficos e Relatorio
    # ------------------------------------------------------------------
    print("[5/5] Gerando graficos e relatorio...")
    plot_comparative_roc(results_trad, results_tb, REPORT_DIR / "roc_tradeability.png")
    plot_pr_curves(results_trad, results_tb, REPORT_DIR / "pr_tradeability.png")
    plot_calibration_comparison(results_trad, results_tb, REPORT_DIR / "calibration_tradeability.png")
    plot_shap_top_features(shap_trad, REPORT_DIR / "shap_tradeability.png")
    plot_fold_metrics(results_trad, results_tb, REPORT_DIR / "fold_metrics_tradeability.png")

    auc_pass = results_trad["overall"]["auc"] >= 0.70
    report = generate_report(results_trad, results_tb, shap_trad, auc_pass, REPORT_MD)

    # Salvar metricas JSON
    trad_metrics = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "001_tradeability",
        "target": "0=Neutro, 1=Tradeable",
        "overall": results_trad["overall"],
        "folds": results_trad["fold_metrics"],
        "criterion_auc_070": {
            "required": 0.70,
            "observed": results_trad["overall"]["auc"],
            "passed": auc_pass,
        },
    }
    (REPORT_DIR / "tradeability_metrics.json").write_text(
        json.dumps(trad_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("RESULTADO FINAL")
    print("=" * 70)
    print(f"  Tradeability AUC: {results_trad['overall']['auc']:.4f}")
    print(f"  TB Direction AUC: {results_tb['overall']['auc']:.4f}")
    print(f"  Delta:            {results_trad['overall']['auc'] - results_tb['overall']['auc']:+.4f}")
    print(f"  Criterio AUC>=0.70: {'APROVADO' if auc_pass else 'REPROVADO'}")
    print(f"  Relatorio: {REPORT_MD}")
    print("=" * 70)


if __name__ == "__main__":
    main()
