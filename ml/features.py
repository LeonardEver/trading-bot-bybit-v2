import numpy as np
import pandas as pd

FRACDIFF_D = 0.4
FRACDIFF_THRESHOLD = 1e-4


def _default_ml_features() -> list[str]:
    from ml.config import FEATURES

    return list(FEATURES)


def apply_strict_feature_lag(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    periods: int = 1,
) -> pd.DataFrame:
    """
    Lag the ML feature view so row t only contains information confirmed by t-1.

    OHLCV, labels, and operational columns are intentionally left untouched; only
    the model feature columns are shifted. Missing feature columns are ignored so
    callers can apply this safely before every optional data source is available.
    """
    if periods < 1:
        raise ValueError("periods must be >= 1 for strict causal feature lag")

    df = df.copy()
    columns = feature_columns if feature_columns is not None else _default_ml_features()
    existing = [col for col in columns if col in df.columns]
    if existing:
        df[existing] = df[existing].shift(periods)
    return df


def get_fractional_diff_weights(d: float = FRACDIFF_D, threshold: float = FRACDIFF_THRESHOLD) -> np.ndarray:
    """
    Return fixed-width fractional differencing weights.
    A value of d between 0 and 1 keeps part of long memory while reducing trend.
    """
    weights = [1.0]
    k = 1
    while True:
        weight = -weights[-1] * (d - k + 1) / k
        if abs(weight) < threshold:
            break
        weights.append(weight)
        k += 1

    return np.array(weights[::-1], dtype=float)


def fractional_difference(
    series: pd.Series,
    d: float = FRACDIFF_D,
    threshold: float = FRACDIFF_THRESHOLD,
) -> pd.Series:
    """Apply fixed-width fractional differencing to a numeric series."""
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    weights = get_fractional_diff_weights(d=d, threshold=threshold)
    width = len(weights)

    if len(clean) < width:
        return pd.Series(np.nan, index=series.index)

    values = clean.to_numpy(dtype=float)
    output = np.full(len(values), np.nan, dtype=float)
    for idx in range(width - 1, len(values)):
        window = values[idx - width + 1:idx + 1]
        if np.isnan(window).any():
            continue
        output[idx] = np.dot(weights, window)

    return pd.Series(output, index=series.index)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate OHLCV, stationary ML features, and operational indicators.
    Nominal price indicators are kept for strategy/risk logic; ml.config.FEATURES
    selects the stationary columns used by the model.
    """
    df = df.copy()

    close = pd.to_numeric(df["close"], errors="coerce").replace(0, np.nan)
    open_ = pd.to_numeric(df["open"], errors="coerce").replace(0, np.nan)
    high = pd.to_numeric(df["high"], errors="coerce").replace(0, np.nan)
    low = pd.to_numeric(df["low"], errors="coerce").replace(0, np.nan)
    volume = pd.to_numeric(df["volume"], errors="coerce").replace(0, np.nan)

    log_close = np.log(close)
    log_volume = np.log(volume)

    # Stationary price and volume features for ML.
    df["log_return"] = log_close.diff()
    df["log_return_3"] = log_close.diff(3)
    df["log_return_5"] = log_close.diff(5)
    df["log_return_15"] = log_close.diff(15)
    df["close_open_log_return"] = np.log(close / open_)
    df["high_low_log_range"] = np.log(high / low)
    df["volume_log_change"] = log_volume.diff()
    df["volume_ma_ratio"] = volume / volume.rolling(window=20).mean()
    df["fracdiff_close"] = fractional_difference(log_close)
    df["fracdiff_close_5"] = df["fracdiff_close"].diff(5)

    # Nominal EMAs kept for technical strategy and operational logs.
    df["ema_20"] = close.ewm(span=20, adjust=False).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False).mean()
    df["ema_200"] = close.ewm(span=200, adjust=False).mean()

    ema_20_log = log_close.ewm(span=20, adjust=False).mean()
    ema_50_log = log_close.ewm(span=50, adjust=False).mean()
    ema_200_log = log_close.ewm(span=200, adjust=False).mean()
    df["ema_20_log_distance"] = log_close - ema_20_log
    df["ema_50_log_distance"] = log_close - ema_50_log
    df["ema_200_log_distance"] = log_close - ema_200_log
    df["ema_20_return"] = df["log_return"].ewm(span=20, adjust=False).mean()
    df["ema_50_return"] = df["log_return"].ewm(span=50, adjust=False).mean()
    df["ema_200_return"] = df["log_return"].ewm(span=200, adjust=False).mean()

    # RSI.
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD on nominal price, plus stationary MACD on log returns for ML.
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    ema12_ret = df["log_return"].ewm(span=12, adjust=False).mean()
    ema26_ret = df["log_return"].ewm(span=26, adjust=False).mean()
    df["macd_return"] = ema12_ret - ema26_ret
    df["macd_signal_return"] = df["macd_return"].ewm(span=9, adjust=False).mean()
    df["macd_hist_return"] = df["macd_return"] - df["macd_signal_return"]

    # Bollinger width normalized by the moving average.
    ma20 = close.rolling(window=20).mean()
    std20 = close.rolling(window=20).std()
    upper = ma20 + (2 * std20)
    lower = ma20 - (2 * std20)
    df["bb_width"] = (upper - lower) / ma20

    # ATR and normalized ATR.
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr"] / close

    # Nominal volume average kept for strategy/risk logic.
    df["volume_ma"] = volume.rolling(window=20).mean()

    if "timestamp" in df.columns:
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            timestamp = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        else:
            timestamp = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        timestamp = pd.Series(pd.NaT, index=df.index)

    typical_price = (high + low + close) / 3
    pv = typical_price * volume
    session_key = timestamp.dt.strftime("%Y-%m-%d").fillna("unknown")
    session_pv = pv.groupby(session_key).cumsum()
    session_volume = volume.groupby(session_key).cumsum()
    anchored_vwap = session_pv / session_volume.replace(0, np.nan)
    df["anchored_vwap"] = anchored_vwap
    df["anchored_vwap_distance"] = (close - anchored_vwap) / close
    df["session_volume_share"] = volume / session_volume.replace(0, np.nan)
    vwap_distance = close - anchored_vwap
    vwap_std = vwap_distance.rolling(window=50).std()
    df["vwap_deviation_zscore"] = vwap_distance / vwap_std.replace(0, np.nan)

    # Extra defaults.
    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = 0.0
    if "timestamp" in df.columns:
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            timestamp = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        else:
            timestamp = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour"] = timestamp.dt.hour
        df["minute"] = timestamp.dt.minute
    else:
        df["hour"] = 0
        df["minute"] = 0
    df["risk_level_encoded"] = df.get("risk_level_encoded", 1)
    df["ml_probability"] = df.get("ml_probability", 0.5)

    # Derivatives defaults; live code overwrites them with Bybit values.
    df["funding_rate"] = df.get("funding_rate", 0.0)
    df["predicted_funding_rate"] = df.get("predicted_funding_rate", df["funding_rate"])
    df["premium_index"] = df.get("premium_index", 0.0)
    df["premium_basis_pct"] = df.get("premium_basis_pct", 0.0)
    df["funding_rate_delta"] = pd.to_numeric(df["funding_rate"], errors="coerce").diff().fillna(0.0)
    df["cvd"] = df.get("cvd", 0.0)
    df["cvd_ratio"] = df.get("cvd_ratio", 0.0)
    df["oi"] = df.get("oi", 0.0)
    df["oi_change_pct"] = df.get("oi_change_pct", 0.0)
    df["liquidation_imbalance"] = df.get("liquidation_imbalance", 0.0)
    df["liquidation_notional"] = df.get("liquidation_notional", 0.0)
    df["liquidation_reversal_signal"] = df.get("liquidation_reversal_signal", 0)
    df["liquidation_cluster_density"] = df.get("liquidation_cluster_density", 0.0)
    df["liquidation_cluster_side"] = df.get("liquidation_cluster_side", 0.0)
    df["spot_cvd_ratio"] = df.get("spot_cvd_ratio", df["cvd_ratio"])
    df["perp_cvd_ratio"] = df.get("perp_cvd_ratio", df["cvd_ratio"])
    df["spot_perp_cvd_divergence"] = df.get(
        "spot_perp_cvd_divergence",
        pd.to_numeric(df["spot_cvd_ratio"], errors="coerce") - pd.to_numeric(df["perp_cvd_ratio"], errors="coerce"),
    )

    return df
