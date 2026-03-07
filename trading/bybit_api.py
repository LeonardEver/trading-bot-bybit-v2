# trading/bybit_api.py
from pybit.unified_trading import HTTP
import os
from dotenv import load_dotenv

load_dotenv()

session = HTTP(
    testnet=os.getenv("BYBIT_TESTNET", "false").lower() == "true",
    demo=os.getenv("BYBIT_DEMO", "false").lower() == "true",
    api_key=os.getenv("BYBIT_API_KEY"),
    api_secret=os.getenv("BYBIT_API_SECRET"),
)

def place_order(symbol, side, qty, take_profit=None, stop_loss=None, trailing_stop=None):
    order_data = {
        "category": "linear",
        "symbol": symbol,
        "side": side.capitalize(),
        "orderType": "Market",
        "qty": qty,
        "timeInForce": "GoodTillCancel"
    }

    if take_profit:
        order_data["takeProfit"] = str(take_profit)
    if stop_loss:
        order_data["stopLoss"] = str(stop_loss)
    if trailing_stop:
        order_data["trailingStop"] = str(trailing_stop)

    try:
        return session.place_order(**order_data)
    except Exception as e:
        return {"error": str(e)}

def get_last_price(symbol):
    try:
        result = session.get_tickers(category="linear", symbol=symbol)
        return float(result["result"]["list"][0]["lastPrice"])
    except Exception as e:
        print(f"Erro ao obter preço: {e}")
        return None

def get_balance():
    try:
        result = session.get_wallet_balance(accountType="UNIFIED")
        accounts = result.get("result", {}).get("list", [])

        if not accounts:
            return 0.0

        for coin in accounts[0].get("coin", []):
            if coin.get("coin") == "USDT":
                return float(coin.get("walletBalance", 0.0))
        return 0.0
    except Exception as e:
        print(f"Erro ao obter saldo: {e}")
        return 0.0

def get_position(symbol):
    try:
        result = session.get_positions(category="linear", symbol=symbol)
        if result["result"]["list"]:
            return result["result"]["list"][0]
        return None
    except Exception as e:
        print(f"Erro ao obter posição: {e}")
        return None

def close_position(symbol, side):
    """Fecha posição ativa enviando ordem oposta."""
    try:
        if side == "Buy":
            opposite = "Sell"
        else:
            opposite = "Buy"

        pos = get_position(symbol)
        if not pos or float(pos.get("size", 0)) <= 0:
            return {"error": "Nenhuma posição aberta para fechar."}

        qty = pos.get("size")

        close_order = {
            "category": "linear",
            "symbol": symbol,
            "side": opposite,
            "orderType": "Market",
            "qty": qty,
            "reduceOnly": True  # garante que fecha a posição
        }

        return session.place_order(**close_order)
    except Exception as e:
        return {"error": str(e)}
    
def get_all_positions(symbol):
    """Retorna todas as posições abertas para o símbolo especificado."""
    try:
        result = session.get_positions(category="linear", symbol=symbol)
        if result.get("retCode") != 0:
            print(f"[ERRO] Não foi possível obter posições: {result}")
            return []

        positions = []
        for pos in result["result"]["list"]:
            if float(pos.get("size", 0)) > 0:
                positions.append({
                    "side": pos.get("side"),
                    "avgPrice": float(pos.get("avgPrice", 0)),
                    "size": float(pos.get("size", 0))
                })

        return positions
    except Exception as e:
        print(f"[ERRO] get_all_positions: {e}")
        return []
    
