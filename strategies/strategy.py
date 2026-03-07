# strategy.py

import pandas as pd
from utils.technical_indicators import calculate_indicators as add_indicators


def check_trade_signal(df: pd.DataFrame) -> str:
    """
    Analisa os indicadores técnicos e retorna uma sugestão de operação.
    Retorna:
        - "buy" se houver sinal de compra
        - "sell" se houver sinal de venda
        - "hold" se não houver sinal claro
    """
    df = add_indicators(df)

    if len(df) < 2:
        return "hold"

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Condição de tendência de alta
    is_uptrend = last["ema20"] > last["ema50"] > last["ema200"]

    # Pullback na EMA 20 ou EMA 50
    pullback_zone = last["close"] > last["ema20"] and prev["close"] < prev["ema20"]

    # Confirmação por RSI e volume
    rsi_ok = last["rsi"] > 50
    volume_ok = last["volume"] > df["volume"].rolling(20).mean().iloc[-1]

    if is_uptrend and pullback_zone and rsi_ok and volume_ok:
        return "buy"

    # Condição de tendência de baixa
    is_downtrend = last["ema20"] < last["ema50"] < last["ema200"]
    pullback_down = last["close"] < last["ema20"] and prev["close"] > prev["ema20"]
    rsi_down = last["rsi"] < 50
    volume_down = last["volume"] > df["volume"].rolling(20).mean().iloc[-1]

    if is_downtrend and pullback_down and rsi_down and volume_down:
        return "sell"

    return "hold"

def generate_trade_signal(df):
    """
    Retorna um dict com sinal e confiança baseada em múltiplos indicadores.
    """
    if df.empty or len(df) < 2:
        return {"signal": "hold", "confidence": 50.0}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0
    total_points = 5  # Quantidade de indicadores usados

    # Indicador 1 - EMA curto acima do longo (tendência de alta)
    if last["ema_20"] > last["ema_50"]:
        score += 1

    # Indicador 2 - RSI acima de 50 (força compradora)
    if last["rsi"] > 50:
        score += 1

    # Indicador 3 - Volume acima da média (confirmação de movimento)
    if last["volume"] > df["volume"].rolling(20).mean().iloc[-1]:
        score += 1

    # Indicador 4 - MACD positivo
    if "macd" in df.columns and last["macd"] > 0:
        score += 1

    # Indicador 5 - Candle atual fechando acima da EMA 20
    if last["close"] > last["ema_20"]:
        score += 1

    # Calcular confiança como percentual de acertos
    confidence = round((score / total_points) * 100, 2)

    # Determinar sinal final
    if score >= 4:  # maioria esmagadora de sinais bullish
        signal = "buy"
    elif score <= 1:  # maioria esmagadora bearish
        signal = "sell"
    else:
        signal = "hold"

    return {"signal": signal, "confidence": confidence}
