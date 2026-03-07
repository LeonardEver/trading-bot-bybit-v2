# utils/ohlcv.py
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import requests
import joblib

# ------------------------------------------------------------------
# Garante que a RAIZ do projeto esteja no sys.path
# (funciona mesmo rodando este arquivo de dentro da pasta utils/)
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------
# Imports do seu projeto (agora a raiz está no sys.path)
# ------------------------------------------------------------------
try:
    from ml.config import FEATURES
    from ml.features import prepare_features
except Exception as e:
    print(f"[IMPORT] Falha ao importar FEATURES/prepare_features: {e}")
    print("-> Verifique se existe 'ml/__init__.py' e 'utils/__init__.py' (podem ser vazios).")
    raise

# ==============================
# Normalização de candles
# ==============================
def _normalize_ohlcv_df(df):
    """
    Normaliza retorno em DataFrame com colunas:
    timestamp (datetime), open, high, low, close, volume
    """
    if df is None:
        return pd.DataFrame()

    # Se vier dict (resposta crua da API)
    if isinstance(df, dict):
        if "result" in df:
            res = df["result"]
            if isinstance(res, dict) and "list" in res:
                df = pd.DataFrame(res["list"])
            elif isinstance(res, list):
                df = pd.DataFrame(res)
            else:
                try:
                    df = pd.DataFrame(res)
                except Exception:
                    df = pd.DataFrame()
        elif "data" in df:
            df = pd.DataFrame(df["data"])
        else:
            try:
                df = pd.DataFrame(df)
            except Exception:
                return pd.DataFrame()

    # Se não for DataFrame ou estiver vazio
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame() if not isinstance(df, pd.DataFrame) else df.copy()

    df = df.copy()
    colmap = {}

    # timestamp
    for cand in ["timestamp", "startTime", "start_time", "start_at", "t", "time"]:
        if cand in df.columns:
            colmap[cand] = "timestamp"
            break
    # OHLCV
    for cand in ["open", "openPrice", "o"]:
        if cand in df.columns:
            colmap[cand] = "open"
            break
    for cand in ["high", "highPrice", "h"]:
        if cand in df.columns:
            colmap[cand] = "high"
            break
    for cand in ["low", "lowPrice", "l"]:
        if cand in df.columns:
            colmap[cand] = "low"
            break
    for cand in ["close", "closePrice", "c", "price"]:
        if cand in df.columns:
            colmap[cand] = "close"
            break
    for cand in ["volume", "vol", "v", "qty"]:
        if cand in df.columns:
            colmap[cand] = "volume"
            break

    if colmap:
        df = df.rename(columns=colmap)

    # Se ainda não tiver timestamp, tenta inferir
    if "timestamp" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "timestamp"})
        else:
            first_col = df.columns[0]
            if np.issubdtype(df[first_col].dtype, np.number):
                df = df.rename(columns={first_col: "timestamp"})

    # Converte timestamp -> datetime (corrigindo o FutureWarning)
    if "timestamp" in df.columns:
        try:
            if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
                # primeiro garante numérico
                df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
                # depois converte assumindo milissegundos
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
                # fallback sem unidade caso tudo vire NaT
                if df["timestamp"].isna().all():
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        except Exception:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Garante tipos numéricos
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove linhas ruins
    if "timestamp" in df.columns and "close" in df.columns:
        df = df.dropna(subset=["timestamp", "close"])
    else:
        return pd.DataFrame()

    return df.reset_index(drop=True)

# ==============================
# Pega candles Bybit
# ==============================
def get_ohlcv(symbol="BTCUSDT", interval="5", limit=200):
    """
    Usa a API v5 pública da Bybit.
    interval: "1","3","5","15","30","60","240","D","W","M"
    """
    url = "https://api.bybit.com/v5/market/kline"
    params = {"category": "linear", "symbol": symbol, "interval": str(interval), "limit": limit}

    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("retCode") != 0:
            print(f"Erro ao obter OHLCV de {symbol}: {data}")
            return pd.DataFrame()

        raw = data.get("result", {}).get("list", [])
        if not raw:
            print("Erro: nenhum dado OHLCV retornado.")
            return pd.DataFrame()

        # A API da Bybit retorna: [start, open, high, low, close, volume, turnover]
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        return _normalize_ohlcv_df(df)

    except Exception as e:
        print(f"Erro em get_ohlcv: {e}")
        return pd.DataFrame()

# ==============================
# Predição com modelo treinado
# ==============================
def predict_signal(df):
    model_path = os.path.join("ml", "model_lgb.pkl")
    if not os.path.exists(model_path):
        print("❌ Modelo não encontrado. Rode train_model.py primeiro.")
        return None

    model = joblib.load(model_path)

    # Prepara features
    df = prepare_features(df)

    # Garante que TODAS as features usadas no treino existem
    for f in FEATURES:
        if f not in df.columns:
            print(f"[WARN] Feature ausente no OHLCV: {f} -> preenchendo com 0")
            df[f] = 0.0

    # Reordena colunas para o mesmo formato do treino
    X = df[FEATURES].ffill().fillna(0)

    # Último candle
    x_last = X.iloc[[-1]]
    probs = model.predict(x_last)[0]

    prob_up = probs
    prob_down = 1 - probs

    signal = "NEUTRO"
    if prob_up > 0.55:
        signal = "COMPRA"
    elif prob_down > 0.55:
        signal = "VENDA"

    print(f"\nÚltimo candle: {df.iloc[-1]['timestamp']} | Close={df.iloc[-1]['close']}")
    print(f"📊 Prob. Alta: {prob_up:.2%} | Prob. Queda: {prob_down:.2%}")
    print(f"➡️ Sinal: {signal}\n")

    return signal

if __name__ == "__main__":
    df = get_ohlcv(symbol="BTCUSDT", interval="5", limit=200)

    if not df.empty:
        sinal = predict_signal(df)
        print(f"SINAL FINAL: {sinal}")
    else:
        print("❌ Nenhum dado de OHLCV retornado.")

