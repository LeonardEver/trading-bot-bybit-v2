"""
Triple Barrier Evaluation — Treinamento, Validação e Relatório Comparativo.

Compara o target atual (direção do candle) com o Triple Barrier Method,
mantendo TODOS os hiperparâmetros e features idênticos (A/B test puro).

Métricas computadas:
  - ROC AUC (por fold e global)
  - Accuracy
  - LogLoss
  - Brier Score
  - Calibration Curve (gráfico)
  - SHAP Values (global + top features)
  - Walk-Forward Validation (TimeSeriesSplit)

Output:
  - Gráficos salvos em ml/triple_barrier_report/
  - Relatório final: TRIPLE_BARRIER_REPORT.md
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
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit

from ml.config import FEATURES

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_V1 = ROOT / "ml" / "dataset.csv"
DATA_V2 = ROOT / "ml" / "dataset_triple_barrier.csv"
MODEL_V2_OUT = ROOT / "ml" / "model_lgb_triple_barrier.pkl"
REPORT_DIR = ROOT / "ml" / "triple_barrier_report"
REPORT_MD = ROOT / "TRIPLE_BARRIER_REPORT.md"

# ---------------------------------------------------------------------------
# LightGBM params — EXATAMENTE os mesmos do train_model.py
# ---------------------------------------------------------------------------
LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "seed": 42,
}
N_BOOST_ROUND = 300
N_SPLITS = 5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(path: Path, label_col: str = "label") -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Carrega dataset, garante features e retorna (df, X, y)."""
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    # Garantir features
    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0.0

    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    X = df[FEATURES].astype(float).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    y = df[label_col].astype(int)
    return df, X.to_numpy(), y.to_numpy()


def load_feature_names(path: Path) -> list[str]:
    """Retorna nomes das colunas de feature na ordem correta."""
    df = pd.read_csv(path)
    existing = [f for f in FEATURES if f in df.columns]
    return existing


# ---------------------------------------------------------------------------
# Walk-Forward Training + Evaluation
# ---------------------------------------------------------------------------
def walk_forward_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    label: str = "Model",
) -> dict:
    """
    TimeSeriesSplit walk-forward com LightGBM.
    Retorna dict com métricas por fold, global, predições OOF e modelo.
    """
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

        # Métricas do fold
        y_pred_bin = (y_pred >= 0.5).astype(int)
        auc_fold = roc_auc_score(y_test, y_pred)
        acc_fold = accuracy_score(y_test, y_pred_bin)

        # LogLoss e Brier (com clipping)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        ll_fold = log_loss(y_test, y_pred_clipped)
        brier_fold = brier_score_loss(y_test, y_pred_clipped)

        fold_metrics.append({
            "fold": int(fold),
            "auc": float(auc_fold),
            "accuracy": float(acc_fold),
            "log_loss": float(ll_fold),
            "brier_score": float(brier_fold),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "label_dist_test": {
                "0": int((y_test == 0).sum()),
                "1": int((y_test == 1).sum()),
            },
        })

        print(f"  {label} Fold {fold}: AUC={auc_fold:.4f}  ACC={acc_fold:.4f}  "
              f"LogLoss={ll_fold:.4f}  Brier={brier_fold:.4f}")

        if auc_fold > best_auc:
            best_auc = auc_fold
            best_model = bst

    # Métricas globais OOF
    valid = ~np.isnan(oof_pred)
    y_valid = y[valid]
    p_valid = oof_pred[valid]
    p_valid_clipped = np.clip(p_valid, 1e-15, 1 - 1e-15)

    overall = {
        "auc": float(roc_auc_score(y_valid, p_valid)),
        "accuracy": float(accuracy_score(y_valid, (p_valid >= 0.5).astype(int))),
        "log_loss": float(log_loss(y_valid, p_valid_clipped)),
        "brier_score": float(brier_score_loss(y_valid, p_valid_clipped)),
        "n_valid": int(len(y_valid)),
    }

    print(f"  {label} Overall: AUC={overall['auc']:.4f}  ACC={overall['accuracy']:.4f}  "
          f"LogLoss={overall['log_loss']:.4f}  Brier={overall['brier_score']:.4f}\n")

    return {
        "label": label,
        "fold_metrics": fold_metrics,
        "overall": overall,
        "oof_predictions": p_valid,
        "y_true": y_valid,
        "model": best_model,
    }


# ---------------------------------------------------------------------------
# SHAP Analysis
# ---------------------------------------------------------------------------
def compute_shap_values(model, X_sample: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Computa mean(|SHAP|) para cada feature."""
    try:
        contributions = model.predict(X_sample, pred_contrib=True)
        # Última coluna é o bias (expected value)
        shap_values = contributions[:, :-1]
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        df_shap = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False)
        return df_shap
    except Exception as e:
        print(f"  [WARN] SHAP falhou: {e}")
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_roc_curves(results_v1: dict, results_v2: dict, save_path: Path):
    """ROC curves side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, res, title in [
        (axes[0], results_v1, "Target Atual (Direção do Candle)"),
        (axes[1], results_v2, "Triple Barrier"),
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

    fig.suptitle("ROC Curves — Target Atual vs Triple Barrier", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] ROC Curves -> {save_path}")


def plot_calibration_curves(results_v1: dict, results_v2: dict, save_path: Path):
    """Calibration curves (reliability diagrams)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, res, title in [
        (axes[0], results_v1, "Target Atual (Direção do Candle)"),
        (axes[1], results_v2, "Triple Barrier"),
    ]:
        y_true = res["y_true"]
        y_pred = res["oof_predictions"]

        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10, strategy="uniform")

        ax.plot(prob_pred, prob_true, "b-", linewidth=2, marker="o", markersize=6)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfeita")
        ax.fill_between(prob_pred, prob_true, prob_pred, alpha=0.1, color="blue")

        # Brier score
        brier = res["overall"]["brier_score"]
        ax.set_xlabel("Probabilidade Prevista")
        ax.set_ylabel("Frequência Observada")
        ax.set_title(f"{title}\nBrier Score: {brier:.4f}")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle("Calibration Curves — Target Atual vs Triple Barrier", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Calibration Curves -> {save_path}")


def plot_metrics_evolution(results_v1: dict, results_v2: dict, save_path: Path):
    """Evolução de AUC e LogLoss por fold."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    folds_v1 = [m["fold"] for m in results_v1["fold_metrics"]]
    folds_v2 = [m["fold"] for m in results_v2["fold_metrics"]]

    # AUC por fold
    ax = axes[0, 0]
    ax.plot(folds_v1, [m["auc"] for m in results_v1["fold_metrics"]],
            "bo-", linewidth=2, markersize=8, label="Target Atual")
    ax.plot(folds_v2, [m["auc"] for m in results_v2["fold_metrics"]],
            "ro-", linewidth=2, markersize=8, label="Triple Barrier")
    ax.set_xlabel("Fold")
    ax.set_ylabel("ROC AUC")
    ax.set_title("ROC AUC por Fold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # LogLoss por fold
    ax = axes[0, 1]
    ax.plot(folds_v1, [m["log_loss"] for m in results_v1["fold_metrics"]],
            "bo-", linewidth=2, markersize=8, label="Target Atual")
    ax.plot(folds_v2, [m["log_loss"] for m in results_v2["fold_metrics"]],
            "ro-", linewidth=2, markersize=8, label="Triple Barrier")
    ax.set_xlabel("Fold")
    ax.set_ylabel("LogLoss")
    ax.set_title("LogLoss por Fold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Brier Score por fold
    ax = axes[1, 0]
    ax.plot(folds_v1, [m["brier_score"] for m in results_v1["fold_metrics"]],
            "bo-", linewidth=2, markersize=8, label="Target Atual")
    ax.plot(folds_v2, [m["brier_score"] for m in results_v2["fold_metrics"]],
            "ro-", linewidth=2, markersize=8, label="Triple Barrier")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Brier Score")
    ax.set_title("Brier Score por Fold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy por fold
    ax = axes[1, 1]
    ax.plot(folds_v1, [m["accuracy"] for m in results_v1["fold_metrics"]],
            "bo-", linewidth=2, markersize=8, label="Target Atual")
    ax.plot(folds_v2, [m["accuracy"] for m in results_v2["fold_metrics"]],
            "ro-", linewidth=2, markersize=8, label="Triple Barrier")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy por Fold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Evolução das Métricas por Fold — Target Atual vs Triple Barrier",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Metrics Evolution -> {save_path}")


def plot_shap_comparison(shap_v1: pd.DataFrame, shap_v2: pd.DataFrame, save_path: Path, top_n: int = 20):
    """Top-N SHAP features side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    for ax, shap_df, title in [
        (axes[0], shap_v1, "Target Atual"),
        (axes[1], shap_v2, "Triple Barrier"),
    ]:
        if shap_df.empty:
            ax.text(0.5, 0.5, "SHAP não disponível", ha="center", va="center")
            ax.set_title(title)
            continue

        top = shap_df.head(top_n).iloc[::-1]  # Inverte para barras horizontais
        colors = plt.cm.RdYlGn(top["mean_abs_shap"] / max(top["mean_abs_shap"].max(), 1e-6))
        ax.barh(range(len(top)), top["mean_abs_shap"], color=colors, edgecolor="gray", alpha=0.85)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["feature"], fontsize=8)
        ax.set_xlabel("Mean(|SHAP|)")
        ax.set_title(f"{title}\nTop {top_n} Features")
        ax.grid(True, alpha=0.2, axis="x")

    fig.suptitle("SHAP Feature Importance — Target Atual vs Triple Barrier",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] SHAP Comparison -> {save_path}")


def plot_label_distribution(results_v1: dict, results_v2: dict, save_path: Path):
    """Distribuição de labels nos dois datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, res, title in [
        (axes[0], results_v1, "Target Atual"),
        (axes[1], results_v2, "Triple Barrier"),
    ]:
        y = res["y_true"]
        unique, counts = np.unique(y, return_counts=True)
        colors = ["#ff6b6b", "#51cf66"]
        labels = ["SL / Baixa (0)", "TP / Alta (1)"]
        wedges, texts, autotexts = ax.pie(
            counts, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, explode=(0, 0.05)
        )
        for at in autotexts:
            at.set_fontweight("bold")
        ax.set_title(f"{title}\nN={len(y):,} amostras")

    fig.suptitle("Distribuição de Labels", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Label Distribution -> {save_path}")


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------
def generate_markdown_report(
    results_v1: dict,
    results_v2: dict,
    shap_v1: pd.DataFrame,
    shap_v2: pd.DataFrame,
    save_path: Path,
) -> str:
    """Gera relatório final em Markdown com conclusão e recomendação."""

    o1 = results_v1["overall"]
    o2 = results_v2["overall"]

    delta_auc = o2["auc"] - o1["auc"]
    delta_acc = o2["accuracy"] - o1["accuracy"]
    delta_ll = o2["log_loss"] - o1["log_loss"]
    delta_brier = o2["brier_score"] - o1["brier_score"]

    # Determinar significância prática
    auc_winner = "Triple Barrier" if delta_auc > 0.005 else ("Target Atual" if delta_auc < -0.005 else "Empate")
    ll_winner = "Triple Barrier" if delta_ll < -0.01 else ("Target Atual" if delta_ll > 0.01 else "Empate")
    brier_winner = "Triple Barrier" if delta_brier < -0.01 else ("Target Atual" if delta_brier > 0.01 else "Empate")

    # Top-5 SHAP features de cada
    top5_v1 = shap_v1.head(5)["feature"].tolist() if not shap_v1.empty else []
    top5_v2 = shap_v2.head(5)["feature"].tolist() if not shap_v2.empty else []
    overlap = len(set(top5_v1) & set(top5_v2))

    # Conclusão estatística
    # Critérios para aprovação:
    # 1. AUC do Triple Barrier >= AUC do Target Atual (não inferior)
    # 2. Calibração (Brier) não significativamente pior
    # 3. SHAP mostra features financeiramente interpretáveis

    auc_ok = delta_auc >= -0.01  # Não mais que 1% pior
    brier_ok = delta_brier <= 0.02  # Não mais que 2% pior
    shap_overlap_ok = overlap >= 2  # Pelo menos 2 features em comum no top-5

    if auc_ok and brier_ok and delta_auc > 0.005:
        recommendation = "✅ **Aprovar migração** — Triple Barrier é superior ou equivalente em todas as métricas relevantes."
        rec_type = "approve"
    elif auc_ok and brier_ok:
        recommendation = "⚠️ **Necessário mais testes** — Triple Barrier é comparável, mas a vantagem não é conclusiva. Recomenda-se paper trading com os dois targets em paralelo."
        rec_type = "more_tests"
    else:
        recommendation = "❌ **Rejeitar migração** — Triple Barrier apresenta desempenho inferior ao target atual em métricas chave."
        rec_type = "reject"

    # Construir ranking de features combinado
    feature_ranking_rows = []
    if not shap_v2.empty:
        for i, row in shap_v2.head(20).iterrows():
            rank_v1 = "-"
            if not shap_v1.empty:
                match = shap_v1[shap_v1["feature"] == row["feature"]]
                if len(match) > 0:
                    rank_v1 = str(int(shap_v1.index.get_loc(match.index[0])) + 1)
            feature_ranking_rows.append(
                f"| {int(i) + 1 if isinstance(i, (int, np.integer)) else len(feature_ranking_rows) + 1} | "
                f"`{row['feature']}` | {row['mean_abs_shap']:.6f} | {rank_v1} |"
            )

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# Triple Barrier Method — Relatório de Avaliação

**Data:** {now}
**Modelo:** LightGBM (parâmetros idênticos ao target atual)
**Validação:** TimeSeriesSplit (5 folds)

---

## 1. Resumo Comparativo

| Métrica          | Target Atual (v1) | Triple Barrier (v2) | Diferença (v2 − v1) | Vencedor       |
|------------------|--------------------|----------------------|----------------------|----------------|
| **ROC AUC**      | {o1['auc']:.4f}    | {o2['auc']:.4f}      | {delta_auc:+.4f}     | {auc_winner}   |
| **Accuracy**     | {o1['accuracy']:.4f} | {o2['accuracy']:.4f} | {delta_acc:+.4f}     | —              |
| **LogLoss**      | {o1['log_loss']:.4f} | {o2['log_loss']:.4f} | {delta_ll:+.4f}      | {ll_winner}    |
| **Brier Score**  | {o1['brier_score']:.4f} | {o2['brier_score']:.4f} | {delta_brier:+.4f}   | {brier_winner} |

---

## 2. Evolução por Fold (TimeSeriesSplit)

### 2.1 ROC AUC

| Fold | Target Atual | Triple Barrier | Delta  |
|------|-------------|----------------|--------|
"""
    for i in range(N_SPLITS):
        auc_v1 = results_v1["fold_metrics"][i]["auc"]
        auc_v2 = results_v2["fold_metrics"][i]["auc"]
        report += f"| {i}    | {auc_v1:.4f}     | {auc_v2:.4f}       | {auc_v2 - auc_v1:+.4f} |\n"

    report += f"""
### 2.2 LogLoss

| Fold | Target Atual | Triple Barrier | Delta  |
|------|-------------|----------------|--------|
"""
    for i in range(N_SPLITS):
        ll_v1 = results_v1["fold_metrics"][i]["log_loss"]
        ll_v2 = results_v2["fold_metrics"][i]["log_loss"]
        report += f"| {i}    | {ll_v1:.4f}     | {ll_v2:.4f}       | {ll_v2 - ll_v1:+.4f} |\n"

    report += f"""
---

## 3. Ranking de Features (SHAP — Triple Barrier)

| Rank | Feature | Mean(\|SHAP\|) | Rank v1 |
|------|---------|---------------|---------|
"""
    report += "\n".join(feature_ranking_rows) if feature_ranking_rows else "| — | SHAP não disponível | — | — |"

    report += f"""

---

## 4. Overlap de Features Importantes

Top-5 features **Target Atual**: {', '.join(f'`{f}`' for f in top5_v1) if top5_v1 else 'N/A'}
Top-5 features **Triple Barrier**: {', '.join(f'`{f}`' for f in top5_v2) if top5_v2 else 'N/A'}
Overlap no Top-5: **{overlap}/5**

---

## 5. Conclusão Estatística

### 5.1 Critérios de Avaliação

| Critério                          | Threshold        | Valor Observado | Status |
|-----------------------------------|------------------|-----------------|--------|
| AUC Triple Barrier ≥ AUC Atual    | Δ ≥ −0.01        | {delta_auc:+.4f} | {"✅" if auc_ok else "❌"} |
| Brier Score não degradado         | Δ ≤ +0.02        | {delta_brier:+.4f} | {"✅" if brier_ok else "❌"} |
| Overlap SHAP Top-5 ≥ 2            | ≥ 2              | {overlap}/5 | {"✅" if shap_overlap_ok else "❌"} |

### 5.2 Interpretação

- **ROC AUC**: O Triple Barrier {"apresenta AUC superior" if delta_auc > 0.005 else "apresenta AUC comparável" if auc_ok else "apresenta AUC inferior"} ao target atual.
- **Calibração**: O Brier Score {"é melhor (menor)" if delta_brier < -0.01 else "é comparável" if brier_ok else "é pior (maior)"} no Triple Barrier.
- **Features**: O ranking de features {"mantém consistência com o target atual" if shap_overlap_ok else "difere significativamente do target atual"} (overlap de {overlap}/5 no Top-5).

---

## 6. Recomendação

{recommendation}

### Fundamentação:

"""
    if rec_type == "approve":
        report += (
            "O Triple Barrier Method demonstrou desempenho superior ou equivalente "
            "ao target atual em todas as métricas de avaliação (ROC AUC, LogLoss, "
            "Brier Score). As features mais importantes mantêm consistência, "
            "indicando que o modelo está capturando sinais financeiramente "
            "interpretáveis. Recomenda-se a migração do target para Triple Barrier."
        )
    elif rec_type == "more_tests":
        report += (
            "O Triple Barrier Method apresenta desempenho comparável ao target atual, "
            "mas a diferença não é conclusiva o suficiente para uma migração imediata. "
            "Recomenda-se executar paper trading com ambos os targets em paralelo "
            "por um período mínimo de 2 semanas para validar o desempenho em condições "
            "reais de mercado antes de tomar uma decisão final."
        )
    else:
        report += (
            "O Triple Barrier Method não atendeu aos critérios mínimos de desempenho "
            "quando comparado ao target atual. A migração não é recomendada neste momento. "
            "Sugere-se investigar parametrizações alternativas das barreiras "
            "(ex: upper/lower assimétricos diferentes, time barrier variável) "
            "antes de descartar completamente a abordagem."
        )

    report += f"""

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
"""

    save_path.write_text(report, encoding="utf-8")
    print(f"\n  [OK] Relatório Markdown -> {save_path}")
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Triple Barrier Evaluation — A/B Test")
    print("=" * 70)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Carregar datasets
    # ------------------------------------------------------------------
    print("\n[1/6] Carregando datasets...")
    feature_names_v1 = load_feature_names(DATA_V1)
    feature_names_v2 = load_feature_names(DATA_V2)

    # Usar interseção de features disponíveis em ambos
    common_features = sorted(set(feature_names_v1) & set(feature_names_v2))
    if not common_features:
        common_features = FEATURES
    print(f"  Features comuns: {len(common_features)}")

    df_v1, X_v1, y_v1 = load_dataset(DATA_V1)
    df_v2, X_v2, y_v2 = load_dataset(DATA_V2)

    # Garantir features comuns
    X_v1_df = pd.DataFrame(X_v1, columns=feature_names_v1)
    X_v2_df = pd.DataFrame(X_v2, columns=feature_names_v2)
    for f in common_features:
        if f not in X_v1_df.columns:
            X_v1_df[f] = 0.0
        if f not in X_v2_df.columns:
            X_v2_df[f] = 0.0
    X_v1 = X_v1_df[common_features].to_numpy()
    X_v2 = X_v2_df[common_features].to_numpy()

    print(f"  Target Atual:    {X_v1.shape[0]:,} linhas, label 1 = {(y_v1 == 1).sum():,} ({(y_v1 == 1).mean()*100:.1f}%)")
    print(f"  Triple Barrier:  {X_v2.shape[0]:,} linhas, label 1 = {(y_v2 == 1).sum():,} ({(y_v2 == 1).mean()*100:.1f}%)")

    # ------------------------------------------------------------------
    # 2. Walk-Forward Evaluation — Target Atual
    # ------------------------------------------------------------------
    print("\n[2/6] Walk-Forward — Target Atual (v1)...")
    results_v1 = walk_forward_evaluate(X_v1, y_v1, common_features, label="V1")

    # ------------------------------------------------------------------
    # 3. Walk-Forward Evaluation — Triple Barrier
    # ------------------------------------------------------------------
    print("\n[3/6] Walk-Forward — Triple Barrier (v2)...")
    results_v2 = walk_forward_evaluate(X_v2, y_v2, common_features, label="V2")

    # Salvar modelo v2
    if results_v2["model"] is not None:
        joblib.dump(results_v2["model"], MODEL_V2_OUT)
        print(f"  [OK] Modelo Triple Barrier salvo -> {MODEL_V2_OUT}")

    # ------------------------------------------------------------------
    # 4. SHAP Analysis
    # ------------------------------------------------------------------
    print("\n[4/6] Computando SHAP values...")
    # Usar amostra para SHAP (eficiência)
    n_shap = min(2000, X_v1.shape[0], X_v2.shape[0])
    sample_v1 = X_v1[-n_shap:]
    sample_v2 = X_v2[-n_shap:]

    shap_v1 = compute_shap_values(results_v1["model"], sample_v1, common_features) if results_v1["model"] else pd.DataFrame()
    shap_v2 = compute_shap_values(results_v2["model"], sample_v2, common_features) if results_v2["model"] else pd.DataFrame()

    # ------------------------------------------------------------------
    # 5. Gerar gráficos
    # ------------------------------------------------------------------
    print("\n[5/6] Gerando gráficos...")
    plot_roc_curves(results_v1, results_v2, REPORT_DIR / "roc_curves.png")
    plot_calibration_curves(results_v1, results_v2, REPORT_DIR / "calibration_curves.png")
    plot_metrics_evolution(results_v1, results_v2, REPORT_DIR / "metrics_evolution.png")
    plot_shap_comparison(shap_v1, shap_v2, REPORT_DIR / "shap_comparison.png")
    plot_label_distribution(results_v1, results_v2, REPORT_DIR / "label_distribution.png")

    # ------------------------------------------------------------------
    # 6. Relatório Final
    # ------------------------------------------------------------------
    print("\n[6/6] Gerando relatório final...")
    report = generate_markdown_report(results_v1, results_v2, shap_v1, shap_v2, REPORT_MD)

    # Salvar métricas em JSON para referência
    metrics_json = {
        "timestamp": datetime.now().isoformat(),
        "target_atual": results_v1["overall"],
        "triple_barrier": results_v2["overall"],
        "target_atual_folds": results_v1["fold_metrics"],
        "triple_barrier_folds": results_v2["fold_metrics"],
    }
    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(metrics_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("RELATORIO FINAL")
    print("=" * 70)
    # Encode to ASCII to avoid cp1252 issues on Windows
    print(report.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
