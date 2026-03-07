# trading/position_mode.py

from pybit.unified_trading import HTTP
import os
from dotenv import load_dotenv

load_dotenv()

session = HTTP(
    api_key=os.getenv("BYBIT_API_KEY"),
    api_secret=os.getenv("BYBIT_API_SECRET"),
    testnet=os.getenv("BYBIT_TESTNET", "false").lower() == "true"
)

def ensure_hedge_mode(symbol: str) -> bool:
    """
    Verifica se o modo hedge está ativado. Se não estiver, tenta ativar.
    """
    try:
        response = session.get_positions(category="linear", symbol=symbol)

        if "result" not in response or "list" not in response["result"]:
            print("Erro ao obter lista de posições.")
            return False

        # Verifica se há mais de uma posição (long e short) para esse símbolo
        position_list = response["result"]["list"]
        if len(position_list) == 2:
            print("Modo hedge já está ativo.")
            return True
        else:
            print("⚠️ Aparentemente o modo hedge não está ativado.")
            print("🔧 Ative o modo hedge manualmente no painel da Bybit.")
            return False

    except Exception as e:
        print(f"Erro ao verificar ou alterar modo de posição: {e}")
        return False
