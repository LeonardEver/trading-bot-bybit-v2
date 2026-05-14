# ml/ml_data_pipeline.py 
"""
Pipeline de dados:
- baixa OHLCV (usa sua função get_ohlcv)
- calcula indicadores (usa calculate_indicators)
- cria label: 1 se TP atingido antes do SL dentro do horizonte H (minutos), senão 0
- salva dataset.csv pronto para treinar
"""
# --- permitir imports a partir da raiz do projeto ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # pasta raiz do projeto
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import os
from utils.ohlcv import get_ohlcv
from ml.features import prepare_features
from sentiment.sentiment_analysis import get_news_sentiment
from ml.config import FEATURES
from ml.features import prepare_features


# Parâmetros
SYMBOL = "BTCUSDT"
TF = "5"                # timeframe base (ajuste se quiser: '1m','5m','15m')
HORIZON_MIN = 15         # horizonte para o label (minutos)
TP_ATR_MULT = 0.8
SL_ATR_MULT = 0.6
LOOKBACK = 5000          # quantos candles buscar (ajuste)
OUT_CSV = Path("ml/dataset.csv")

def fetch_ohlcv(symbol, interval=None, limit=LOOKBACK):
    """
    Tenta chamar get_ohlcv com assinatura cheia ou a versão simples.
    Retorna DataFrame padronizado com coluna 'timestamp' (datetime).
    """
    try:
        df = get_ohlcv(symbol, interval=interval, limit=limit)
    except TypeError:
        df = get_ohlcv(symbol)
    except Exception as e:
        print("Erro ao obter OHLCV de", symbol, ":", e)
        return pd.DataFrame()

    if df is None:
        return pd.DataFrame()

    # debug: mostrar colunas brutas para ajudar diagnosis
    print("[DEBUG] Colunas OHLCV brutas:", list(df.columns) if isinstance(df, pd.DataFrame) else "nao-DF")

    # garantir timestamp/datatypes — se get_ohlcv já normalizou via utils/ohlcv, isso será OK
    if isinstance(df, pd.DataFrame):
        if "timestamp" not in df.columns:
            # tenta nomes comuns
            if "startTime" in df.columns:
                df = df.rename(columns={"startTime": "timestamp"})
            elif "start_time" in df.columns:
                df = df.rename(columns={"start_time": "timestamp"})
            elif df.index.__class__.__name__ == "DatetimeIndex":
                df = df.reset_index().rename(columns={"index": "timestamp"})

        # converter se numérico
        if "timestamp" in df.columns:
            if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
                try:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
                    if df["timestamp"].isna().all():
                        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                except Exception:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        # se não for DataFrame, tentar construir um
        try:
            df = pd.DataFrame(df)
        except Exception:
            return pd.DataFrame()

    # se não temos timestamp válido, retorna vazio para não quebrar pipeline
    if "timestamp" not in df.columns or df["timestamp"].isna().all():
        print("[DEBUG] timestamp ausente ou inválido após normalização. Colunas:", list(df.columns))
        return pd.DataFrame()

    return df

def map_sentiment_to_score(label):
    # adapta seu get_news_sentiment: 'positive','neutral','negative'
    if label is None: return 0.0
    l = str(label).lower()
    if "pos" in l: return 1.0
    if "neg" in l: return -1.0
    return 0.0


def build_labels(df, horizon_min=HORIZON_MIN, tp_mult=TP_ATR_MULT, sl_mult=SL_ATR_MULT):
    df = df.copy().reset_index(drop=True)
    n = len(df)
    labels = np.zeros(n, dtype=int)

    # níveis de TP/SL baseados em ATR
    if "atr" not in df.columns:
        raise RuntimeError("ATR não encontrado — rode calculate_indicators antes.")
    tp_levels = df["close"] + df["atr"] * tp_mult
    sl_levels = df["close"] - df["atr"] * sl_mult

    times = pd.to_datetime(df["timestamp"])
    # calculamos horizon_time para cada linha e varremos à frente
    for i in range(n):
        start = times.iloc[i]
        horizon_time = start + pd.Timedelta(minutes=horizon_min)
        tp = tp_levels.iloc[i]
        sl = sl_levels.iloc[i]
        j = i + 1
        # percorre candles até o horizonte
        while j < n and times.iloc[j] <= horizon_time:
            high = df["high"].iloc[j]
            low = df["low"].iloc[j]
            # se ambos atingidos no mesmo candle --> marca 0 (conservador)
            if high >= tp and low <= sl:
                labels[i] = 0
                break
            if high >= tp:
                labels[i] = 1
                break
            if low <= sl:
                labels[i] = 0
                break
            j += 1
        # senão continua 0
    df["label"] = labels
    return df


def assign_risk_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Atribui nível de risco baseado na confiabilidade do sinal (sentiment_score).
    """
    conditions = [
        (df["sentiment_score"] >= 0.8),   # risco baixo
        (df["sentiment_score"] >= 0.6),   # risco médio
        (df["sentiment_score"] < 0.6),    # risco alto
    ]
    choices = ["low", "medium", "high"]

    df["risk_level"] = np.select(conditions, choices, default="medium")
    return df


def attach_sentiment(df):
    # Atenção: chamada ao get_news_sentiment pode ser lenta — aqui fazemos chamada por candle.
    # Melhor coleta em lote histórico para produção. Aqui é simples.
    sent_scores = []
    for ts in pd.to_datetime(df["timestamp"]):
        # sua função recebe "BTC" no exemplo; mapeie como preferir
        try:
            s = get_news_sentiment("BTC")  # adaptável: pode usar intervalo de tempo em função
        except Exception:
            s = None
        sent_scores.append(map_sentiment_to_score(s))
    df["sentiment_score"] = sent_scores
    return df


def main():
    print("Baixando OHLCV...")
    df = fetch_ohlcv(SYMBOL, interval=TF, limit=LOOKBACK)
    if df.empty:
        print("Erro: nenhum dado OHLCV retornado.")
        return

    # Garantir coluna timestamp
    if "startTime" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"startTime": "timestamp"})

    # Converter para datetime se ainda for numérico
    if np.issubdtype(df["timestamp"].dtype, np.number):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # 🔹 Calcular indicadores antes de labels
    print("Calculando features ML (FracDiff, LogReturns, etc)...")
    try:
        df = prepare_features(df)
    except Exception as e:
        print("Erro ao calcular features:", e)
        return

    # 🔹 Anexar sentimento
    print("Anexando sentimento (pode demorar)...")
    try:
        df = attach_sentiment(df)
    except Exception as e:
        print("Erro no sentiment attach:", e)
        df["sentiment_score"] = 0.0

    # 🔹 Construir labels (agora já tem ATR, RSI, etc.)
    print("Construindo labels...")
    df = build_labels(df)

    # 🔹 Atribui nível de Risco
    print("Atribuindo nível de risco...")
    df = assign_risk_level(df)

    # 🔹 Salvar dataset
    print("Preparando features estacionarias...")
    df = prepare_features(df)

    print(f"Salvando dataset em {OUT_CSV}")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print("✅ Concluído.")


if __name__ == "__main__":
    main()
