from trading.bybit_api import get_balance, get_position


# trading/risk_management.py
def calculate_order_qty(symbol, risk_level, price):
    """
    Calcula quantidade de contratos baseado no risco.
    """
    saldo_usdt = 100.0  # ou buscar da API get_balance('USDT')
    if risk_level == "baixo":
        pct = 0.1
    elif risk_level == "medio":
        pct = 0.25
    else:
        pct = 0.5

    valor_usdt = saldo_usdt * pct
    qty = valor_usdt / price  # converte em BTC

    # garante mínimo exigido pela Bybit
    if qty < 0.001:
        qty = 0.001

    return round(qty, 3)  # ou 4 casas, depende do tick size



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

    if direction == "Buy":
        stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
        take_profit_price = entry_price * (1 + take_profit_pct / 100)

        if current_price <= stop_loss_price:
            should_exit, reason = True, "Stop Loss atingido"
        elif current_price >= take_profit_price:
            should_exit, reason = True, "Take Profit atingido"
        elif current_price >= entry_price * (1 + trailing_trigger_pct / 100):
            new_stop_loss_price = current_price * (1 - trailing_step_pct / 100)
            if new_stop_loss_price > stop_loss_price:
                stop_loss_price, reason = new_stop_loss_price, "Trailing Stop ajustado"

    elif direction == "Sell":
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
