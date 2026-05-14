# strategy.py

import pandas as pd
from utils.technical_indicators import calculate_indicators as add_indicators

MAX_LONG_FUNDING_RATE = 0.0005
MAX_LONG_PREDICTED_FUNDING_RATE = 0.0007


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

# strategies/strategy.py

def funding_allows_long(derivatives_metrics=None) -> bool:
    """Avoid new longs when the perpetual carry cost is unusually expensive."""
    if not derivatives_metrics:
        return True

    funding_rate = float(derivatives_metrics.get("funding_rate") or 0.0)
    predicted_funding = float(derivatives_metrics.get("predicted_funding_rate") or funding_rate)
    return (
        funding_rate <= MAX_LONG_FUNDING_RATE
        and predicted_funding <= MAX_LONG_PREDICTED_FUNDING_RATE
    )


def generate_trade_signal(df, derivatives_metrics=None):
    """
    Retorna um dict com sinal e confiança baseada em múltiplos indicadores.
    Sistema de pontuação: +1 (Bullish) / -1 (Bearish)
    """
    if df.empty or len(df) < 2:
        return {"signal": "hold", "confidence": 0.0}

    last = df.iloc[-1]
    score = 0.0
    max_score = 8.0  # Quantidade de indicadores avaliados

    # Indicador 1 - EMA cruzamento
    if last["ema_20"] > last["ema_50"]:
        score += 1
    elif last["ema_20"] < last["ema_50"]:
        score -= 1

    # Indicador 2 - RSI
    if last["rsi"] > 55:
        score += 1
    elif last["rsi"] < 45:
        score -= 1

    # Indicador 3 - Volume
    volume_ma = df["volume"].rolling(20).mean().iloc[-1]
    if last["volume"] > volume_ma:
        # Só dá o ponto se o volume alto acompanhou a tendência do preço
        if last["close"] > last["open"]:
            score += 1
        else:
            score -= 1

    # Indicador 4 - MACD
    if "macd" in df.columns:
        if last["macd"] > 0 and last["macd"] > last["macd_signal"]:
            score += 1
        elif last["macd"] < 0 and last["macd"] < last["macd_signal"]:
            score -= 1

    # Indicador 5 - Preço vs EMA 20
    if last["close"] > last["ema_20"]:
        score += 1
    elif last["close"] < last["ema_20"]:
        score -= 1

    # Indicador 6 - CVD / agressao de fluxo
    if "cvd_ratio" in df.columns:
        if last["cvd_ratio"] > 0.10:
            score += 1
        elif last["cvd_ratio"] < -0.10:
            score -= 1

    # Indicador 7 - Open Interest confirma movimento com capital novo
    if "oi_change_pct" in df.columns:
        price_direction = 1 if last["close"] > last["open"] else -1
        if last["oi_change_pct"] > 0.002:
            score += price_direction
        elif last["oi_change_pct"] < -0.002:
            score -= price_direction * 0.5

    # Indicador 8 - Cascatas de liquidacao sugerem reversao a media
    if "liquidation_reversal_signal" in df.columns:
        score += float(last["liquidation_reversal_signal"])

    # ----------------------------------------------------
    # O CÁLCULO MÁGICO DA CONFIANÇA
    # ----------------------------------------------------
    confidence = (abs(score) / max_score) * 100

    if score >= 2:
        signal = "buy"
    elif score <= -2:
        signal = "sell"
    else:
        signal = "hold"

    if signal == "buy" and not funding_allows_long(derivatives_metrics):
        signal = "hold"
        confidence = 0.0

    return {"signal": signal, "confidence": round(confidence, 1)}
