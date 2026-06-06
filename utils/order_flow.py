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
    """Attach latest order-flow metrics to the newest row only.

    Older candles must keep their own historical snapshot. Broadcasting the
    current tape/OI/liquidation state into the past creates look-ahead leakage.
    """
    df = df.copy()
    cvd_metrics = cvd_metrics or {}
    open_interest_metrics = open_interest_metrics or {}
    liquidation_metrics = liquidation_metrics or {}

    updates = {
        "cvd": float(cvd_metrics.get("cvd", 0.0) or 0.0),
        "cvd_ratio": float(cvd_metrics.get("cvd_ratio", 0.0) or 0.0),
        "oi": float(open_interest_metrics.get("open_interest", 0.0) or 0.0),
        "oi_change_pct": float(open_interest_metrics.get("oi_change_pct", 0.0) or 0.0),
        "liquidation_imbalance": float(liquidation_metrics.get("liquidation_imbalance", 0.0) or 0.0),
        "liquidation_notional": float(liquidation_metrics.get("liquidation_notional", 0.0) or 0.0),
        "liquidation_reversal_signal": int(liquidation_metrics.get("liquidation_reversal_signal", 0) or 0),
        "liquidation_cluster_density": float(liquidation_metrics.get("liquidation_cluster_density", 0.0) or 0.0),
        "liquidation_cluster_side": float(liquidation_metrics.get("liquidation_cluster_side", 0.0) or 0.0),
        "spot_cvd_ratio": float(cvd_metrics.get("spot_cvd_ratio", cvd_metrics.get("cvd_ratio", 0.0)) or 0.0),
        "perp_cvd_ratio": float(cvd_metrics.get("perp_cvd_ratio", cvd_metrics.get("cvd_ratio", 0.0)) or 0.0),
        "spot_perp_cvd_divergence": float(cvd_metrics.get("spot_perp_cvd_divergence", 0.0) or 0.0),
    }

    for col in updates:
        if col not in df.columns:
            df[col] = np.nan

    if not df.empty:
        last_idx = df.index[-1]
        for col, value in updates.items():
            df.at[last_idx, col] = value

    return df


def calculate_liquidation_metrics(events, lookback=100, imbalance_threshold=0.65) -> dict:
    """Summarize liquidation cascades into an imbalance and mean-reversion hint."""
    recent = list(events or [])[-lookback:]
    long_liq = 0.0
    short_liq = 0.0
    cluster_buckets = {}

    for event in recent:
        side = str(event.get("side", "")).lower()
        price = float(event.get("price") or event.get("p") or 0.0)
        qty = float(event.get("qty") or event.get("size") or event.get("v") or 0.0)
        notional = abs(price * qty)
        if price > 0 and notional > 0:
            bucket = round(price / 100) * 100
            if bucket not in cluster_buckets:
                cluster_buckets[bucket] = {"long": 0.0, "short": 0.0}
            if side == "sell":
                cluster_buckets[bucket]["long"] += notional
            elif side == "buy":
                cluster_buckets[bucket]["short"] += notional
        if side == "sell":
            long_liq += notional
        elif side == "buy":
            short_liq += notional

    total = long_liq + short_liq
    imbalance = (short_liq - long_liq) / total if total else 0.0
    cluster_density = 0.0
    cluster_side = 0.0
    if total and cluster_buckets:
        dominant = max(cluster_buckets.values(), key=lambda item: item["long"] + item["short"])
        dominant_total = dominant["long"] + dominant["short"]
        cluster_density = dominant_total / total
        cluster_side = (dominant["short"] - dominant["long"]) / dominant_total if dominant_total else 0.0

    reversal_signal = 0
    if total > 0 and abs(imbalance) >= imbalance_threshold:
        reversal_signal = -1 if imbalance > 0 else 1

    return {
        "long_liquidation_notional": long_liq,
        "short_liquidation_notional": short_liq,
        "liquidation_notional": total,
        "liquidation_imbalance": imbalance,
        "liquidation_reversal_signal": reversal_signal,
        "liquidation_cluster_density": cluster_density,
        "liquidation_cluster_side": cluster_side,
    }


def calculate_spot_perp_cvd_divergence(spot_trades, perp_trades) -> dict:
    """Compare spot and perpetual CVD pressure."""
    spot = calculate_cvd_from_trades(spot_trades)
    perp = calculate_cvd_from_trades(perp_trades)
    divergence = float(spot.get("cvd_ratio", 0.0) - perp.get("cvd_ratio", 0.0))
    return {
        "spot_cvd_ratio": float(spot.get("cvd_ratio", 0.0)),
        "perp_cvd_ratio": float(perp.get("cvd_ratio", 0.0)),
        "spot_perp_cvd_divergence": divergence,
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
