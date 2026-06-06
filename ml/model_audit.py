import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from ml.config import FEATURES


DATA_CSV = ROOT / "ml" / "dataset.csv"
AUDIT_DIR = ROOT / "ml" / "audit"
CALIBRATOR_OUT = ROOT / "ml" / "model_calibrator.pkl"


def load_xy(path: Path = DATA_CSV):
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    max_rows = int(os.getenv("AUDIT_MAX_ROWS", "0") or 0)
    if max_rows > 0 and len(df) > max_rows:
        df = df.tail(max_rows).reset_index(drop=True)
    for feature in FEATURES:
        if feature not in df.columns:
            df[feature] = 0.0
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    x = df[FEATURES].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    y = df["label"].astype(int)
    return df, x, y


def _auc(y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_pred))


def profit_metrics(y_true, y_pred, threshold=0.5):
    signals = np.where(y_pred >= threshold, 1, -1)
    realized = np.where(np.asarray(y_true) == 1, 1, -1)
    pnl = signals * realized
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) else float("inf")
    equity = np.cumsum(pnl)
    drawdown = equity - np.maximum.accumulate(equity)
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
    sharpe = float(pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 else 0.0
    return {"profit_factor": profit_factor, "max_drawdown": max_drawdown, "sharpe": sharpe}


def walk_forward_predictions(x, y, n_splits=5):
    splitter = TimeSeriesSplit(n_splits=n_splits)
    oof = pd.Series(np.nan, index=x.index, dtype=float)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(x)):
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42 + fold,
            verbose=-1,
        )
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict_proba(x.iloc[test_idx])[:, 1]
        oof.iloc[test_idx] = pred
        fold_metrics.append(
            {
                "fold": fold,
                "auc": _auc(y.iloc[test_idx], pred),
                "accuracy": float(accuracy_score(y.iloc[test_idx], pred >= 0.5)),
                **profit_metrics(y.iloc[test_idx], pred),
            }
        )

    valid = oof.notna()
    overall = {
        "auc": _auc(y[valid], oof[valid]),
        "accuracy": float(accuracy_score(y[valid], oof[valid] >= 0.5)),
        "log_loss": float(log_loss(y[valid], np.clip(oof[valid], 1e-6, 1 - 1e-6))),
        **profit_metrics(y[valid], oof[valid]),
    }
    return oof, fold_metrics, overall


def permutation_information_gain(x, y, repeats=3, top_n=None):
    rng = np.random.default_rng(42)
    model = lgb.LGBMClassifier(objective="binary", n_estimators=300, random_state=77, verbose=-1)
    model.fit(x, y)
    base_pred = model.predict_proba(x)[:, 1]
    base_auc = _auc(y, base_pred)
    rows = []
    columns = list(x.columns[:top_n]) if top_n else list(x.columns)

    for column in columns:
        drops = []
        for _ in range(repeats):
            shuffled_x = x.copy()
            shuffled_values = shuffled_x[column].to_numpy().copy()
            rng.shuffle(shuffled_values)
            shuffled_x[column] = shuffled_values
            shuffled_pred = model.predict_proba(shuffled_x)[:, 1]
            drops.append(base_auc - _auc(y, shuffled_pred))
        rows.append({"feature": column, "auc_drop": float(np.mean(drops))})

    return pd.DataFrame(rows).sort_values("auc_drop", ascending=False)


def collinearity_report(x, threshold=0.95):
    corr = x.corr(numeric_only=True).abs()
    rows = []
    for i, left in enumerate(corr.columns):
        for right in corr.columns[i + 1:]:
            value = corr.at[left, right]
            if pd.notna(value) and value >= threshold:
                rows.append({"feature_a": left, "feature_b": right, "abs_corr": float(value)})
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False) if rows else pd.DataFrame(columns=["feature_a", "feature_b", "abs_corr"])


def fit_platt_calibrator(oof_pred, y):
    valid = oof_pred.notna()
    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(oof_pred[valid].to_numpy().reshape(-1, 1), y[valid])
    joblib.dump(calibrator, CALIBRATOR_OUT)
    return calibrator


def export_shap_or_importance(x, y):
    model = lgb.LGBMClassifier(objective="binary", n_estimators=300, random_state=123, verbose=-1)
    model.fit(x, y)
    importance = pd.DataFrame({"feature": x.columns, "importance": model.feature_importances_})
    importance = importance.sort_values("importance", ascending=False)
    importance.to_csv(AUDIT_DIR / "feature_importance.csv", index=False)

    try:
        sample = x.tail(min(len(x), 2000))
        contributions = model.booster_.predict(sample, pred_contrib=True)
        values = contributions[:, :-1]
        shap_summary = pd.DataFrame({"feature": sample.columns, "mean_abs_shap": np.abs(values).mean(axis=0)})
        shap_summary.sort_values("mean_abs_shap", ascending=False).to_csv(AUDIT_DIR / "shap_summary.csv", index=False)
        return "lightgbm_pred_contrib"
    except Exception:
        return "feature_importance"


def run_audit():
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    _, x, y = load_xy()
    oof, folds, overall = walk_forward_predictions(x, y)
    valid = oof.notna()
    info_gain = permutation_information_gain(x.loc[valid], y.loc[valid], top_n=None)
    collinear = collinearity_report(x)
    fit_platt_calibrator(oof, y)
    explanation_method = export_shap_or_importance(x, y)

    info_gain.to_csv(AUDIT_DIR / "permutation_information_gain.csv", index=False)
    collinear.to_csv(AUDIT_DIR / "collinearity_report.csv", index=False)
    pd.DataFrame(folds).to_csv(AUDIT_DIR / "walk_forward_folds.csv", index=False)

    baseline_pred = np.full(valid.sum(), y.loc[valid].mean())
    baseline = {
        "auc": _auc(y.loc[valid], baseline_pred),
        "accuracy": float(accuracy_score(y.loc[valid], baseline_pred >= 0.5)),
        **profit_metrics(y.loc[valid], baseline_pred),
    }
    report = {
        "overall": overall,
        "baseline": baseline,
        "kpi_improved": bool(overall["auc"] > baseline["auc"] or overall["sharpe"] > baseline["sharpe"]),
        "positive_information_gain_features": int((info_gain["auc_drop"] > 0).sum()),
        "collinearity_pairs_over_095": int(len(collinear)),
        "explanation_method": explanation_method,
        "calibrator_path": str(CALIBRATOR_OUT),
    }
    (AUDIT_DIR / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_audit(), indent=2))
