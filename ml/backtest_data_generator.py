# ml/backtest_data_generator.py
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.config import FEATURES
from ml.ml_data_pipeline import build_causal_dataset


SYMBOL = "BTCUSDT"
INTERVAL = "15"
CANDLES_TO_FETCH = 200000
DATASET_PATH = os.path.join(ROOT, "ml", "dataset.csv")


def fetch_historical_data(symbol, interval, total_candles):
    """Download historical Bybit candles with pagination."""
    print(f"A descarregar {total_candles} velas historicas de {symbol} ({interval}m)...")
    url = "https://api.bybit.com/v5/market/kline"
    all_klines = []
    end_time = int(time.time() * 1000)

    while len(all_klines) < total_candles:
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": 1000,
            "end": end_time,
        }

        try:
            resp = requests.get(url, params=params, timeout=20).json()
            if "result" not in resp or not resp["result"]["list"]:
                break

            chunk = resp["result"]["list"]
            all_klines.extend(chunk)
            end_time = int(chunk[-1][0]) - 1

            print(f"   -> {len(all_klines)} velas descarregadas...")
            time.sleep(0.1)
        except Exception as e:
            print(f"Erro na API: {e}")
            break

    df = pd.DataFrame(all_klines, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    if df.empty:
        return df

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["timestamp", "close"]).reset_index(drop=True)


def generate_backtest_dataset():
    df = fetch_historical_data(SYMBOL, INTERVAL, CANDLES_TO_FETCH)
    if df.empty:
        print("Nenhum dado historico recuperado.")
        return

    df["sentiment_score"] = 0.0
    df["risk_level"] = "high"
    df["risk_level_encoded"] = 2

    print("A calcular dataset causal com lag estrito...")
    df_dataset = build_causal_dataset(df)
    cols_to_save = [col for col in ["timestamp", *FEATURES, "target_return", "label"] if col in df_dataset.columns]
    df_dataset[cols_to_save].to_csv(DATASET_PATH, index=False)

    print(f"\nDataset causal gerado com {len(df_dataset)} linhas.")
    print(f"Arquivo salvo em: {DATASET_PATH}")


if __name__ == "__main__":
    generate_backtest_dataset()
