# trading/technical_indicators.py

import pandas as pd
import ta  # Technical Analysis library (pip install ta)

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona indicadores técnicos comuns ao DataFrame OHLCV.
    Requer colunas: ['open', 'high', 'low', 'close', 'volume']
    """
    df = df.copy()

    # ===== MÉDIAS MÓVEIS =====
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['ema_200'] = df['close'].ewm(span=200).mean()

    # ===== RSI (Relative Strength Index) =====
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()

    # ===== MACD =====
    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()

    # ===== BANDAS DE BOLLINGER =====
    bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = df['bb_upper'] - df['bb_lower']

    # ===== ATR (Average True Range) =====
    df['atr'] = ta.volatility.AverageTrueRange(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=14
    ).average_true_range()

    # ===== VOLUME MÉDIO =====
    df['volume_ma'] = df['volume'].rolling(window=20).mean()

    # ===== VOLATILIDADE (ATR) =====
    # Mede a amplitude média dos últimos 14 candles
    df['atr'] = ta.volatility.AverageTrueRange(
        high=df['high'], 
        low=df['low'], 
        close=df['close'], 
        window=14
    ).average_true_range()

    return df


# Função auxiliar para detectar "squeeze" (BB dentro da Keltner)
def is_bollinger_squeeze(df: pd.DataFrame) -> pd.Series:
    """
    Retorna uma Series booleana indicando se há um "squeeze" (compressão).
    Squeeze clássico = Bandas de Bollinger dentro do canal de Keltner.
    """
    keltner = ta.volatility.KeltnerChannel(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=20
    )

    squeeze = (
        df['bb_upper'] < keltner.keltner_channel_hband() &
        df['bb_lower'] > keltner.keltner_channel_lband()
    )
    return squeeze.fillna(False)
