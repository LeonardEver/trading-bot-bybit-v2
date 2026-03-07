import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
BASE_URL = "https://api-testnet.bybit.com"  # Para conta real. Use "https://api-testnet.bybit.com" para testes.

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
LEVERAGE = 3  # Configurável
RISK_PER_TRADE = 0.01  # 1%
DAILY_LOSS_LIMIT = 0.05
DAILY_PROFIT_LIMIT = 0.10
INTERVALS = ["15", "30", "60"]  # em minutos

USE_TRAILING_STOP = True
