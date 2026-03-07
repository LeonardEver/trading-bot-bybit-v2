# ml/features.py
import pandas as pd
import numpy as np

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todas as features no DataFrame OHLCV.
    """
    df = df.copy()

    # ===============================
    # Médias móveis exponenciais
    # ===============================
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

    # ===============================
    # RSI
    # ===============================
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ===============================
    # MACD
    # ===============================
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # ===============================
    # Bandas de Bollinger
    # ===============================
    ma20 = df["close"].rolling(window=20).mean()
    std20 = df["close"].rolling(window=20).std()
    upper = ma20 + (2 * std20)
    lower = ma20 - (2 * std20)
    df["bb_width"] = (upper - lower) / ma20

    # ===============================
    # ATR
    # ===============================
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # ===============================
    # Volume média
    # ===============================
    df["volume_ma"] = df["volume"].rolling(window=20).mean()

    # ===============================
    # Placeholders extras
    # ===============================
    df["sentiment_score"] = 0.0
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["risk_level_encoded"] = 1
    df["ml_probability"] = 0.5

    return df
