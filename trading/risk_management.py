from trading.bybit_api import get_balance, get_position

ATR_STOP_MULTIPLIERS = {
    "trending": 0.8,
    "high_vol": 1.2,
    "ranging": 1.0,
    "normal": 1.0,
}

ATR_TAKE_PROFIT_MULTIPLIERS = {
    "trending": 2.0,
    "high_vol": 1.8,
    "ranging": 1.2,
    "normal": 1.5,
}


# trading/risk_management.py
def calculate_order_qty(symbol, risk_level, price):
    """
    Calcula quantidade de contratos baseado no risco (Max 2%).
    """
    saldo_usdt = 100.0  # O ideal é usar a sua função get_balance() da API aqui
    
    # Níveis institucionais de exposição:
    if risk_level == "baixo":
        pct = 0.01   # 1% do capital
    elif risk_level == "medio":
        pct = 0.015  # 1.5% do capital
    else:
        pct = 0.02   # 2% do capital

    valor_usdt = saldo_usdt * pct
    qty = valor_usdt / price  # converte em BTC

    # garante mínimo exigido pela Bybit
    if qty < 0.001:
        qty = 0.001

    return round(qty, 3)


def calculate_atr_exit_prices(price, atr, side, market_regime="normal"):
    """
    Dynamic ATR-based TP/SL. This centralizes stop distance in the risk module.
    """
    price = float(price)
    atr = float(atr)
    stop_mult = ATR_STOP_MULTIPLIERS.get(market_regime, ATR_STOP_MULTIPLIERS["normal"])
    tp_mult = ATR_TAKE_PROFIT_MULTIPLIERS.get(market_regime, ATR_TAKE_PROFIT_MULTIPLIERS["normal"])

    side_normalized = side.title() if isinstance(side, str) else "Buy"
    if side_normalized in {"Buy", "Long"}:
        take_profit = price + (atr * tp_mult)
        stop_loss = price - (atr * stop_mult)
    else:
        take_profit = price - (atr * tp_mult)
        stop_loss = price + (atr * stop_mult)

    return {
        "take_profit": round(take_profit, 1),
        "stop_loss": round(stop_loss, 1),
        "tp_atr_multiple": tp_mult,
        "sl_atr_multiple": stop_mult,
    }


def has_open_position(symbol: str, side: str = None) -> bool:
    """
    Verifica se existe posição aberta para o símbolo.
    Se `side` for 'Buy' ou 'Sell', verifica apenas aquela direção.
    No modo Hedge, a Bybit retorna uma lista de posições (long e short separadas).
    """
    positions = get_position(symbol)

    if not positions:
        return False

    # Se a API retornar apenas um dicionário (modo single)
    if isinstance(positions, dict):
        if float(positions.get("size", 0)) <= 0:
            return False
        if side and positions.get("side") != side:
            return False
        return True

    # Se a API retornar lista (modo hedge)
    for pos in positions:
        if float(pos.get("size", 0)) > 0:
            if not side or pos.get("side") == side:
                return True

    return False


def calculate_exit_conditions(entry_price, current_price, direction,
                               atr=None, market_regime="normal",
                               stop_loss_pct=1.0, take_profit_pct=2.0,
                               trailing_trigger_pct=1.5, trailing_step_pct=0.3):
    """
    Define quando sair da posição com base em SL, TP e Trailing Stop.
    """
    stop_loss_price = None
    take_profit_price = None
    new_stop_loss_price = None
    should_exit = False
    reason = ""

    if atr is not None:
        atr_exits = calculate_atr_exit_prices(entry_price, atr, direction, market_regime)
        stop_loss_price = atr_exits["stop_loss"]
        take_profit_price = atr_exits["take_profit"]
    elif direction == "Buy":
        stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
        take_profit_price = entry_price * (1 + take_profit_pct / 100)

    if direction == "Buy":
        if current_price <= stop_loss_price:
            should_exit, reason = True, "Stop Loss atingido"
        elif current_price >= take_profit_price:
            should_exit, reason = True, "Take Profit atingido"
        elif current_price >= entry_price * (1 + trailing_trigger_pct / 100):
            new_stop_loss_price = current_price * (1 - trailing_step_pct / 100)
            if new_stop_loss_price > stop_loss_price:
                stop_loss_price, reason = new_stop_loss_price, "Trailing Stop ajustado"

    elif direction == "Sell":
        if atr is None:
            stop_loss_price = entry_price * (1 + stop_loss_pct / 100)
            take_profit_price = entry_price * (1 - take_profit_pct / 100)

        if current_price >= stop_loss_price:
            should_exit, reason = True, "Stop Loss atingido"
        elif current_price <= take_profit_price:
            should_exit, reason = True, "Take Profit atingido"
        elif current_price <= entry_price * (1 - trailing_trigger_pct / 100):
            new_stop_loss_price = current_price * (1 + trailing_step_pct / 100)
            if new_stop_loss_price < stop_loss_price:
                stop_loss_price, reason = new_stop_loss_price, "Trailing Stop ajustado"

    return {
        "should_exit": should_exit,
        "stop_loss_price": stop_loss_price,
        "take_profit_price": take_profit_price,
        "reason": reason
    }
