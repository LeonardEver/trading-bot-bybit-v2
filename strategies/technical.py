import pandas as pd
import ta

EMA_PERIODS = [9, 21, 50, 200]

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Garantir consistência dos dados
    df = df.copy()
    df = df.sort_index()

    # Converter para float
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

    # EMA e SMA
    for p in EMA_PERIODS:
        df[f"EMA{p}"] = df["close"].ewm(span=p, adjust=False).mean()
        df[f"SMA{p}"] = df["close"].rolling(window=p).mean()

    # RSI
    df["RSI"] = ta.momentum.RSIIndicator(close=df["close"]).rsi()

    # MACD
    macd = ta.trend.MACD(close=df["close"])
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df["close"])
    df["BB_MIDDLE"] = bb.bollinger_mavg()
    df["BB_UPPER"] = bb.bollinger_hband()
    df["BB_LOWER"] = bb.bollinger_lband()

    # OBV
    obv = ta.volume.OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"])
    df["OBV"] = obv.on_balance_volume()

    # ADX
    adx = ta.trend.ADXIndicator(high=df["high"], low=df["low"], close=df["close"])
    df["ADX"] = adx.adx()

    return df

def is_confluence_signal(df: pd.DataFrame) -> str:
    last = df.iloc[-1]
    buy_signals = 0
    sell_signals = 0

    # Médias móveis
    if last["EMA9"] > last["EMA21"] > last["EMA50"]:
        buy_signals += 1
    elif last["EMA9"] < last["EMA21"] < last["EMA50"]:
        sell_signals += 1

    # RSI
    if last["RSI"] < 30:
        buy_signals += 1
    elif last["RSI"] > 70:
        sell_signals += 1

    # MACD
    if last["MACD"] > last["MACD_SIGNAL"]:
        buy_signals += 1
    elif last["MACD"] < last["MACD_SIGNAL"]:
        sell_signals += 1

    # Bollinger Bands
    if last["close"] < last["BB_LOWER"]:
        buy_signals += 1
    elif last["close"] > last["BB_UPPER"]:
        sell_signals += 1

    # ADX
    if last["ADX"] > 25:
        if buy_signals > sell_signals:
            buy_signals += 1
        elif sell_signals > buy_signals:
            sell_signals += 1

    if buy_signals >= 4:
        return "long"
    elif sell_signals >= 4:
        return "short"
    else:
        return None
