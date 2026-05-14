"""
ML feature configuration.

The model consumes stationary transforms only. Nominal price columns such as
close/ema_20/ema_50/ema_200 remain available elsewhere for trading logic, but
are intentionally excluded here to reduce non-stationary leakage.
"""

FEATURES = [
    # Features Estacionárias (O que o modelo realmente deve olhar)
    "log_return", "log_return_3", "log_return_5", "log_return_15",
    "close_open_log_return", "high_low_log_range", "volume_log_change",
    "volume_ma_ratio", "fracdiff_close", "fracdiff_close_5",
    "ema_20_return", "ema_50_return", "ema_200_return",
    "macd_return", "macd_signal_return", "macd_hist_return",
    "atr_pct",
    
    # Derivativos / Order Flow (Mesmo que mockados no momento)
    "funding_rate", "predicted_funding_rate", "premium_index",
    "premium_basis_pct", "cvd_ratio", "oi_change_pct",
    "liquidation_imbalance", "liquidation_notional", "liquidation_reversal_signal",
    
    # Contexto Temporal / Sentimento
    "sentiment_score", "hour", "minute", "risk_level_encoded"
]