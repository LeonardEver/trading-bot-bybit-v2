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
    recv_window=60000,
)

def place_order(
    symbol,
    side,
    qty,
    take_profit=None,
    stop_loss=None,
    trailing_stop=None,
    order_type="Market",
    price=None,
    time_in_force="GoodTillCancel",
):
    order_data = {
        "category": "linear",
        "symbol": symbol,
        "side": side.capitalize(),
        "orderType": order_type,
        "qty": qty,
        "timeInForce": time_in_force
    }

    if price is not None:
        order_data["price"] = str(price)

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


def get_orderbook(symbol, limit=25):
    try:
        result = session.get_orderbook(category="linear", symbol=symbol, limit=limit)
        book = result.get("result", {})
        bids = book.get("b", []) or book.get("bids", [])
        asks = book.get("a", []) or book.get("asks", [])
        return {
            "bids": [(float(level[0]), float(level[1])) for level in bids],
            "asks": [(float(level[0]), float(level[1])) for level in asks],
        }
    except Exception as e:
        print(f"Erro ao obter book: {e}")
        return {"bids": [], "asks": []}


def get_best_bid_ask(symbol):
    book = get_orderbook(symbol, limit=1)
    best_bid = book["bids"][0][0] if book["bids"] else None
    best_ask = book["asks"][0][0] if book["asks"] else None
    return best_bid, best_ask


def get_maker_price(symbol, side, tick_size=0.1, max_spread_pct=0.0005):
    best_bid, best_ask = get_best_bid_ask(symbol)
    if best_bid is None or best_ask is None:
        return None

    mid = (best_bid + best_ask) / 2
    spread_pct = (best_ask - best_bid) / mid if mid else 0.0
    if spread_pct > max_spread_pct:
        return None

    if side.capitalize() == "Buy":
        return round(best_bid, 1)
    return round(best_ask, 1)


def slippage_within_threshold(reference_price, execution_price, side, max_slippage_pct=0.0008):
    if not reference_price or not execution_price:
        return False

    reference_price = float(reference_price)
    execution_price = float(execution_price)
    if side.capitalize() == "Buy":
        slippage = (execution_price - reference_price) / reference_price
    else:
        slippage = (reference_price - execution_price) / reference_price
    return slippage <= max_slippage_pct


def place_maker_order(symbol, side, qty, take_profit=None, stop_loss=None, trailing_stop=None, price=None):
    maker_price = price if price is not None else get_maker_price(symbol, side)
    if maker_price is None:
        return {"error": "Spread acima do limite ou book indisponivel para ordem maker."}

    return place_order(
        symbol=symbol,
        side=side,
        qty=qty,
        take_profit=take_profit,
        stop_loss=stop_loss,
        trailing_stop=trailing_stop,
        order_type="Limit",
        price=maker_price,
        time_in_force="PostOnly",
    )


def cancel_order(symbol, order_id=None, order_link_id=None):
    try:
        params = {"category": "linear", "symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        if order_link_id:
            params["orderLinkId"] = order_link_id
        return session.cancel_order(**params)
    except Exception as e:
        return {"error": str(e)}


def get_last_price(symbol):
    try:
        result = session.get_tickers(category="linear", symbol=symbol)
        return float(result["result"]["list"][0]["lastPrice"])
    except Exception as e:
        print(f"Erro ao obter preço: {e}")
        return None

def get_derivatives_metrics(symbol):
    """
    Return Bybit linear contract metrics used by carry-cost and premium filters.
    Values default to 0.0 so the strategy can continue if Bybit omits a field.
    """
    try:
        result = session.get_tickers(category="linear", symbol=symbol)
        ticker = result.get("result", {}).get("list", [{}])[0]

        last_price = float(ticker.get("lastPrice") or 0.0)
        mark_price = float(ticker.get("markPrice") or last_price or 0.0)
        index_price = float(ticker.get("indexPrice") or 0.0)
        funding_rate = float(ticker.get("fundingRate") or 0.0)
        predicted_funding = float(
            ticker.get("predictedFundingRate")
            or ticker.get("nextFundingRate")
            or funding_rate
            or 0.0
        )
        premium_basis_pct = ((mark_price - index_price) / index_price) if index_price else 0.0

        return {
            "funding_rate": funding_rate,
            "predicted_funding_rate": predicted_funding,
            "next_funding_time": ticker.get("nextFundingTime"),
            "mark_price": mark_price,
            "index_price": index_price,
            "premium_index": premium_basis_pct,
            "premium_basis_pct": premium_basis_pct,
        }
    except Exception as e:
        print(f"Erro ao obter metricas de derivativos: {e}")
        return {
            "funding_rate": 0.0,
            "predicted_funding_rate": 0.0,
            "next_funding_time": None,
            "mark_price": 0.0,
            "index_price": 0.0,
            "premium_index": 0.0,
            "premium_basis_pct": 0.0,
        }


def get_public_trades(symbol, limit=200):
    try:
        result = session.get_public_trade_history(category="linear", symbol=symbol, limit=limit)
        return result.get("result", {}).get("list", [])
    except Exception as e:
        print(f"Erro ao obter trades publicos: {e}")
        return []


def get_open_interest_metrics(symbol, interval_time="5min", limit=2):
    try:
        result = session.get_open_interest(
            category="linear",
            symbol=symbol,
            intervalTime=interval_time,
            limit=limit,
        )
        rows = result.get("result", {}).get("list", [])
        if not rows:
            return {"open_interest": 0.0, "oi_change_pct": 0.0}

        current = float(rows[0].get("openInterest") or 0.0)
        previous = float(rows[1].get("openInterest") or current) if len(rows) > 1 else current
        change_pct = ((current - previous) / previous) if previous else 0.0
        return {"open_interest": current, "oi_change_pct": change_pct}
    except Exception as e:
        print(f"Erro ao obter open interest: {e}")
        return {"open_interest": 0.0, "oi_change_pct": 0.0}


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
    
