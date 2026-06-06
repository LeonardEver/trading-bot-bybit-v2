# ml/ml_data_pipeline.py
"""
Pipeline de dados causal:
- baixa OHLCV
- anexa dados exogenos brutos quando disponiveis
- calcula indicadores
- cria Y(t) como direcao realizada no candle t
- aplica lag estrito em X para que a linha t use apenas informacao de t-1
- salva dataset.csv e ml/dataset.csv prontos para treino
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from ml.config import FEATURES
from ml.features import apply_strict_feature_lag, prepare_features
from sentiment.sentiment_analysis import get_news_sentiment
from utils.ohlcv import get_ohlcv


SYMBOL = "BTCUSDT"
TF = "5"
LOOKBACK = 100000
OUT_CSVS = [ROOT / "dataset.csv", ROOT / "ml" / "dataset.csv"]


def _timestamp_value_to_ms(value):
    if pd.isna(value):
        return pd.NA

    numeric = pd.to_numeric(value, errors="coerce")
    if not pd.isna(numeric):
        numeric = float(numeric)
        if numeric < 1_000_000_000_000:
            numeric *= 1000
        return int(round(numeric))

    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return pd.NA
    return int(parsed.timestamp() * 1000)


def to_timestamp_ms(series: pd.Series) -> pd.Series:
    """Normalize timestamps to integer milliseconds for internal storage."""
    return series.map(_timestamp_value_to_ms).astype("Int64")


def fetch_ohlcv(symbol, interval=None, limit=LOOKBACK):
    """
    Tenta chamar get_ohlcv com assinatura cheia ou a versao simples.
    Retorna DataFrame padronizado com coluna timestamp em milissegundos.
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

    print("[DEBUG] Colunas OHLCV brutas:", list(df.columns) if isinstance(df, pd.DataFrame) else "nao-DF")

    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return pd.DataFrame()

    df = df.copy()
    if "timestamp" not in df.columns:
        if "startTime" in df.columns:
            df = df.rename(columns={"startTime": "timestamp"})
        elif "start_time" in df.columns:
            df = df.rename(columns={"start_time": "timestamp"})
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "timestamp"})

    if "timestamp" not in df.columns:
        print("[DEBUG] timestamp ausente apos normalizacao. Colunas:", list(df.columns))
        return pd.DataFrame()

    df["timestamp"] = to_timestamp_ms(df["timestamp"])
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        print("[DEBUG] timestamp invalido apos normalizacao. Colunas:", list(df.columns))
        return pd.DataFrame()

    return df.sort_values("timestamp").reset_index(drop=True)


def map_sentiment_to_score(label):
    if label is None:
        return 0.0
    value = str(label).lower()
    if "pos" in value or "bull" in value:
        return 1.0
    if "neg" in value or "bear" in value:
        return -1.0
    return 0.0


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create Y(t): close direction realized during candle t."""
    df = df.copy().reset_index(drop=True)
    close = pd.to_numeric(df["close"], errors="coerce")
    df["target_return"] = close.pct_change()
    df["label"] = np.where(df["target_return"].notna(), (df["target_return"] > 0).astype(int), np.nan)
    return df


def assign_risk_level(df: pd.DataFrame) -> pd.DataFrame:
    """Atribui nivel de risco bruto; o valor de ML sera defasado depois."""
    df = df.copy()
    conditions = [
        (df["sentiment_score"] >= 0.8),
        (df["sentiment_score"] >= 0.6),
        (df["sentiment_score"] < 0.6),
    ]
    choices = ["low", "medium", "high"]
    df["risk_level"] = np.select(conditions, choices, default="medium")
    df["risk_level_encoded"] = df["risk_level"].map({"low": 0, "medium": 1, "high": 2}).fillna(1)
    return df


def attach_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    sent_scores = []
    for _ in pd.to_datetime(df["timestamp"], unit="ms", errors="coerce"):
        try:
            sentiment = get_news_sentiment("BTC")
        except Exception:
            sentiment = None
        sent_scores.append(map_sentiment_to_score(sentiment))
    df = df.copy()
    df["sentiment_score"] = sent_scores
    return df


def build_causal_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["timestamp"] = to_timestamp_ms(df["timestamp"])

    df = prepare_features(df)
    df = build_labels(df)
    df = apply_strict_feature_lag(df, FEATURES, periods=1)
    df = df.dropna(subset=FEATURES + ["label"]).reset_index(drop=True)
    return df


def regenerate_existing_datasets(source_csv: Path | str = ROOT / "dataset.csv") -> pd.DataFrame:
    """Clean existing CSV data and rewrite both training datasets with strict lag."""
    source_csv = Path(source_csv)
    if not source_csv.exists():
        raise FileNotFoundError(f"Dataset fonte nao encontrado: {source_csv}")

    raw_df = pd.read_csv(source_csv)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(raw_df.columns)
    if missing:
        raise ValueError(f"Dataset fonte sem colunas obrigatorias: {sorted(missing)}")

    if "sentiment_score" not in raw_df.columns:
        raw_df["sentiment_score"] = 0.0
    if "risk_level_encoded" not in raw_df.columns:
        raw_df["risk_level_encoded"] = 1

    cleaned = build_causal_dataset(raw_df)
    for out_csv in OUT_CSVS:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(out_csv, index=False)
    return cleaned


def main():
    print("Baixando OHLCV...")
    df = fetch_ohlcv(SYMBOL, interval=TF, limit=LOOKBACK)
    if df.empty:
        print("Erro: nenhum dado OHLCV retornado.")
        return

    print("Anexando sentimento (pode demorar)...")
    try:
        df = attach_sentiment(df)
    except Exception as e:
        print("Erro no sentiment attach:", e)
        df["sentiment_score"] = 0.0

    print("Atribuindo nivel de risco...")
    df = assign_risk_level(df)

    print("Calculando features e labels causais X(t-1) -> Y(t)...")
    try:
        df = build_causal_dataset(df)
    except Exception as e:
        print("Erro ao construir dataset causal:", e)
        return

    for out_csv in OUT_CSVS:
        print(f"Salvando dataset em {out_csv}")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    print("Concluido.")


if __name__ == "__main__":
    main()
