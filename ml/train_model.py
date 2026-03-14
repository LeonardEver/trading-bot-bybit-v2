# ml/train_model.py
"""
Treina um modelo LightGBM com validação temporal (TimeSeriesSplit)
Usa dataset gerado do MongoDB ou CSV (ml/dataset.csv)
Salva o modelo em ml/model_lgb.pkl
"""

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score
from pathlib import Path
import matplotlib.pyplot as plt

DATA_CSV = Path("ml/dataset.csv")
MODEL_OUT = Path("ml/model_lgb.pkl")

# Features usadas pelo modelo
FEATURES = [
    "close","volume","ema_20","ema_50","ema_200",
    "rsi","macd","macd_signal","macd_hist","bb_width","atr","volume_ma",
    "sentiment_score","hour","minute",
    "risk_level_encoded","ml_probability"
]

def load_dataset():
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {DATA_CSV}")

    # 1. Lê o dataset sem forçar a busca pela coluna timestamp
    df = pd.read_csv(DATA_CSV)
    print(f"🔹 Dataset carregado: {df.shape[0]} linhas")

    # 2. Se o timestamp existir (dataset antigo), organiza e descarta
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")
        df = df.drop(columns=["timestamp"])

    # 3. Garantir label
    if "label" not in df.columns:
        print("⚠ Coluna 'label' ausente — criando a partir do PnL")
        if "pnl" in df.columns:
            df["label"] = np.where(df["pnl"] > 0, 1, 0)
        else:
            df["label"] = 0

    df = df.dropna(subset=["label"])

    # 4. Converter risco para número
    if "risk_level" in df.columns:
        risk_map = {"baixo": 0, "medio": 1, "alto": 2}
        df["risk_level_encoded"] = df["risk_level"].map(risk_map).fillna(1)

    # 5. Garantir todas as features
    for f in FEATURES:
        if f not in df.columns:
            print(f"[WARN] Feature ausente: {f} -> preenchendo com 0")
            df[f] = 0.0

    return df

def train():
    df = load_dataset()
    X = df[FEATURES].ffill().fillna(0)
    y = df["label"].astype(int)

    tscv = TimeSeriesSplit(n_splits=5)
    best_model = None
    best_auc = -1
    auc_scores = []
    acc_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "seed": 42
        }
        bst = lgb.train(params, train_data, num_boost_round=300)
        y_pred = bst.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        acc = accuracy_score(y_test, (y_pred >= 0.5).astype(int))

        auc_scores.append(auc)
        acc_scores.append(acc)

        print(f"Fold {fold} → AUC: {auc:.4f} | ACC: {acc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_model = bst

    if best_model is None:
        raise RuntimeError("Falha ao treinar modelo.")

    print(f"\n✅ Treino finalizado — Melhor AUC: {best_auc:.4f}")
    joblib.dump(best_model, MODEL_OUT)
    print(f"📦 Modelo salvo em {MODEL_OUT}")

    # Plotar métricas
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(auc_scores)), auc_scores, marker="o", label="AUC")
    plt.plot(range(len(acc_scores)), acc_scores, marker="x", label="Accuracy")
    plt.title("Métricas por Fold (TimeSeriesSplit)")
    plt.xlabel("Fold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ml/training_metrics.png")
    print("📊 Métricas salvas em ml/training_metrics.png")

if __name__ == "__main__":
    train()
