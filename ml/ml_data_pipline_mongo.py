"""
ml/ml_data_pipeline_mongo.py
Cria dataset a partir dos trades salvos no MongoDB.
Útil para aprendizado contínuo do bot em produção.
"""

import pandas as pd
from pymongo import MongoClient
from pathlib import Path
from trading.logger import log_event

OUT_CSV = Path("ml/dataset_mongo.csv")

# Features usadas no modelo (devem bater com main.py)
FEATURES = [
    "ema_20", "ema_50", "ema_200",
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_width", "atr", "volume", "volume_ma",
    "sentiment_score", "hour", "minute"
]

def get_trades_from_mongo(limit=None):
    """Busca trades reais do MongoDB"""
    try:
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
        db = client["trading_bot"]
        trades = db["trades"]

        cursor = trades.find({}).sort("timestamp", -1)
        if limit:
            cursor = cursor.limit(limit)

        data = list(cursor)
        if not data:
            log_event("[ML_PIPELINE_MONGO] Nenhum trade encontrado no MongoDB.")
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Remove _id do Mongo
        if "_id" in df.columns:
            df.drop(columns=["_id"], inplace=True)

        # Converte datas
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        log_event(f"[ML_PIPELINE_MONGO] {len(df)} trades carregados do MongoDB.")
        return df

    except Exception as e:
        log_event(f"[ML_PIPELINE_MONGO ERRO] Falha ao buscar dados: {e}")
        return pd.DataFrame()

def build_dataset(limit=None, save_csv=True):
    """Gera dataset estruturado a partir dos trades"""
    df = get_trades_from_mongo(limit)
    if df.empty:
        return pd.DataFrame()

    # Cria target com base no PnL real
    df["target"] = (df["pnl"] > 0).astype(int)

    # Garante que só pega colunas relevantes
    cols = [c for c in FEATURES if c in df.columns]
    dataset = df[cols + ["target"]].dropna()

    if save_csv:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(OUT_CSV, index=False)
        log_event(f"[ML_PIPELINE_MONGO] Dataset salvo em {OUT_CSV} ({len(dataset)} linhas).")

    return dataset

if __name__ == "__main__":
    build_dataset()
