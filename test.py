from utils.ohlcv import get_ohlcv
from strategies.technical import calculate_indicators, is_confluence_signal

symbol = "BTCUSDT"
df = get_ohlcv(symbol)
df = calculate_indicators(df)

signal = is_confluence_signal(df)
print(f"Confluência para {symbol}: {signal}")
