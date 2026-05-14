import numpy as np
import pandas as pd


def calculate_cvd_from_trades(trades) -> dict:
    """Calculate cumulative volume delta from Bybit public trades."""
    buy_volume = 0.0
    sell_volume = 0.0

    for trade in trades or []:
        side = str(trade.get("side", "")).lower()
        qty = float(trade.get("size") or trade.get("qty") or trade.get("v") or 0.0)
        if side == "buy":
            buy_volume += qty
        elif side == "sell":
            sell_volume += qty

    cvd = buy_volume - sell_volume
    total_volume = buy_volume + sell_volume
    cvd_ratio = cvd / total_volume if total_volume else 0.0

    return {
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
        "cvd": cvd,
        "cvd_ratio": cvd_ratio,
    }


def add_order_flow_features(
    df: pd.DataFrame,
    cvd_metrics: dict | None = None,
    open_interest_metrics: dict | None = None,
    liquidation_metrics: dict | None = None,
) -> pd.DataFrame:
    """Attach order-flow features to every row so the latest candle carries them into ML/logs."""
    df = df.copy()
    cvd_metrics = cvd_metrics or {}
    open_interest_metrics = open_interest_metrics or {}
    liquidation_metrics = liquidation_metrics or {}

    df["cvd"] = float(cvd_metrics.get("cvd", 0.0) or 0.0)
    df["cvd_ratio"] = float(cvd_metrics.get("cvd_ratio", 0.0) or 0.0)
    df["oi"] = float(open_interest_metrics.get("open_interest", 0.0) or 0.0)
    df["oi_change_pct"] = float(open_interest_metrics.get("oi_change_pct", 0.0) or 0.0)
    df["liquidation_imbalance"] = float(liquidation_metrics.get("liquidation_imbalance", 0.0) or 0.0)
    df["liquidation_notional"] = float(liquidation_metrics.get("liquidation_notional", 0.0) or 0.0)
    df["liquidation_reversal_signal"] = int(liquidation_metrics.get("liquidation_reversal_signal", 0) or 0)

    return df


def calculate_liquidation_metrics(events, lookback=100, imbalance_threshold=0.65) -> dict:
    """Summarize liquidation cascades into an imbalance and mean-reversion hint."""
    recent = list(events or [])[-lookback:]
    long_liq = 0.0
    short_liq = 0.0

    for event in recent:
        side = str(event.get("side", "")).lower()
        price = float(event.get("price") or event.get("p") or 0.0)
        qty = float(event.get("qty") or event.get("size") or event.get("v") or 0.0)
        notional = abs(price * qty)
        if side == "sell":
            long_liq += notional
        elif side == "buy":
            short_liq += notional

    total = long_liq + short_liq
    imbalance = (short_liq - long_liq) / total if total else 0.0

    reversal_signal = 0
    if total > 0 and abs(imbalance) >= imbalance_threshold:
        reversal_signal = -1 if imbalance > 0 else 1

    return {
        "long_liquidation_notional": long_liq,
        "short_liquidation_notional": short_liq,
        "liquidation_notional": total,
        "liquidation_imbalance": imbalance,
        "liquidation_reversal_signal": reversal_signal,
    }


def estimate_cvd_from_candles(df: pd.DataFrame, window=50) -> dict:
    """Fallback CVD proxy when trade tape is unavailable."""
    if df.empty:
        return calculate_cvd_from_trades([])

    recent = df.tail(window).copy()
    signed_volume = np.where(recent["close"] >= recent["open"], recent["volume"], -recent["volume"])
    cvd = float(np.nansum(signed_volume))
    total_volume = float(np.nansum(np.abs(recent["volume"])))

    return {
        "buy_volume": float(np.nansum(np.where(signed_volume > 0, signed_volume, 0.0))),
        "sell_volume": float(np.nansum(np.where(signed_volume < 0, -signed_volume, 0.0))),
        "cvd": cvd,
        "cvd_ratio": cvd / total_volume if total_volume else 0.0,
    }
