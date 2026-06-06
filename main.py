import time
import csv
import joblib
import pandas as pd
import numpy as np
import uuid
import os
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from utils.ohlcv import get_ohlcv
from strategies.strategy import generate_trade_signal
from sentiment.sentiment_analysis import get_news_sentiment
from trading.bybit_api import (
    place_order,
    get_last_price,
    get_all_positions,
    close_position,
    get_balance,
    get_derivatives_metrics,
    get_public_trades,
    get_open_interest_metrics,
    get_maker_price,
    place_maker_order,
    slippage_within_threshold,
    cancel_order,
)
from trading.logger import log_event
from utils.technical_indicators import calculate_indicators
from database.mongo_logger import log_trade, log_signal_decision, update_signal_outcome
from ml.config import FEATURES
from ml.features import apply_strict_feature_lag, prepare_features
from trading.risk_management import calculate_atr_exit_prices
from utils.order_flow import (
    add_order_flow_features,
    calculate_cvd_from_trades,
    calculate_liquidation_metrics,
    calculate_spot_perp_cvd_divergence,
    estimate_cvd_from_candles,
)
from pybit.unified_trading import WebSocket 


print(">>> Script main.py carregado com sucesso")


SYMBOL = "BTCUSDT"
WS_INTERVAL = "5"
LOOP_INTERVAL = 5
LOG_FILE = "trading_log.csv"
WATCHDOG_TIMEOUT_SECONDS = 6 * 60
TESTNET_MODE = False
WEBSOCKET_CHANNEL_TYPE = "linear"
MAX_DAILY_DRAWDOWN_PCT = -4.0
CIRCUIT_BREAKER_HOURS = 24

TAKE_PROFIT_PCT = 0.0010
STOP_LOSS_PCT = 0.0010
TRAILING_STOP_PCT = 0.0008

ultima_ordem = {"side": None, "hora": datetime.min}
ordem_lock = Lock()

# Variáveis Globais de Controle e Proteção
bloqueio_ate = datetime.min
ultima_modificacao_modelo = 0.0
ultimo_candle_recebido = datetime.now()

historico_candles_ws = pd.DataFrame()
candle_cache_lock = Lock()
trade_tape_ws = []
liquidation_events_ws = []
market_flow_lock = Lock()

# Pesos iniciais
peso_tecnico = 0.5
peso_sentimento = 0.5

# Caminho e features do modelo ML
MODEL_PATH = Path("ml/model_lgb.pkl")
ENSEMBLE_PATH = Path("ml/model_ensemble.pkl")
CALIBRATOR_PATH = Path("ml/model_calibrator.pkl")

# Carrega modelo, se existir
model = None
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
    ultima_modificacao_modelo = os.path.getmtime(MODEL_PATH) # Já anota a hora do modelo atual
    log_event(f"Modelo ML carregado de {MODEL_PATH}")
else:
    log_event("⚠ Modelo ML não encontrado — rodando sem filtro de ML.")


def normalize_timestamp_ms_value(value):
    """Return a timestamp as integer milliseconds."""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            value = value.tz_localize("UTC")
        return int(value.timestamp() * 1000)
    if isinstance(value, datetime):
        return int(value.timestamp() * 1000)

    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(parsed):
            return None
        return int(parsed.timestamp() * 1000)

    numeric = float(numeric)
    if numeric < 1_000_000_000_000:
        numeric *= 1000
    return int(round(numeric))


def normalize_timestamp_ms_series(series):
    """Normalize a timestamp Series to nullable integer milliseconds."""
    return series.map(normalize_timestamp_ms_value).astype("Int64")


if ENSEMBLE_PATH.exists():
    model = joblib.load(ENSEMBLE_PATH)
    ultima_modificacao_modelo = os.path.getmtime(ENSEMBLE_PATH)
    log_event(f"Ensemble ML carregado de {ENSEMBLE_PATH}")


# ============================================================
# CORRECTED FUNCTIONS (Priority 1-5)
# ============================================================

def calculate_recent_edge(lookback_trades=50):
    """
    Calculate actual win rate, expectancy, and Sharpe from trading_log.csv
    Returns: (win_rate, avg_win, avg_loss, profit_factor, expectancy, sharpe)
    """
    if not Path(LOG_FILE).exists():
        return 0.33, 0.01, 0.01, 1.0, 0.0, 0.0  # Default conservative
    
    try:
        df_trades = pd.read_csv(LOG_FILE)
        df_trades = df_trades.tail(lookback_trades).copy()
        
        if len(df_trades) < 10:
            return 0.33, 0.01, 0.01, 1.0, 0.0, 0.0
        
        pnls = df_trades['pnl'].astype(float).values
        
        wins = pnls[pnls > 0.001]
        losses = pnls[pnls < -0.001]
        
        n_wins = len(wins)
        n_losses = len(losses)
        n_total = n_wins + n_losses
        
        if n_total == 0:
            return 0.33, 0.01, 0.01, 1.0, 0.0, 0.0
        
        win_rate = n_wins / n_total
        avg_win = wins.mean() if len(wins) > 0 else 0.001
        avg_loss = -losses.mean() if len(losses) > 0 else 0.001
        
        profit_factor = (n_wins * avg_win) / max(n_losses * avg_loss, 0.001)
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        returns = pnls / 100
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        log_event(f"[EDGE METRICS] WR={win_rate:.1%} | PF={profit_factor:.2f} | "
                  f"Exp={expectancy:.4f} | Sharpe={sharpe:.2f}")
        
        return win_rate, avg_win, avg_loss, profit_factor, expectancy, sharpe
        
    except Exception as e:
        log_event(f"[ERROR] Edge calculation: {e}")
        return 0.33, 0.01, 0.01, 1.0, 0.0, 0.0


def calculate_today_pnl_pct():
    """Return today's realized PnL as a percentage of current equity."""
    if not Path(LOG_FILE).exists():
        return 0.0

    try:
        trades = pd.read_csv(LOG_FILE)
        if trades.empty or "timestamp" not in trades.columns or "pnl" not in trades.columns:
            return 0.0

        timestamps = pd.to_datetime(trades["timestamp"], errors="coerce")
        today_mask = timestamps.dt.date == datetime.now().date()
        today_pnl = pd.to_numeric(trades.loc[today_mask, "pnl"], errors="coerce").fillna(0.0).sum()
        equity = float(get_balance() or 0.0)
        if equity <= 0:
            return 0.0
        return (today_pnl / equity) * 100
    except Exception as e:
        log_event(f"[ERROR] Daily drawdown calculation: {e}")
        return 0.0


def enforce_daily_drawdown_circuit_breaker():
    """Pause new entries for 24h if realized daily PnL reaches -4%."""
    global bloqueio_ate

    daily_pnl_pct = calculate_today_pnl_pct()
    if daily_pnl_pct <= MAX_DAILY_DRAWDOWN_PCT:
        bloqueio_ate = max(bloqueio_ate, datetime.now() + timedelta(hours=CIRCUIT_BREAKER_HOURS))
        log_event(
            f"[CIRCUIT] Daily drawdown {daily_pnl_pct:.2f}% <= {MAX_DAILY_DRAWDOWN_PCT:.2f}%. "
            f"Trading paused until {bloqueio_ate.strftime('%Y-%m-%d %H:%M:%S')}."
        )
        return False
    return True


def calculate_order_qty_kelly_criterion(symbol, risk_level, current_price, capital="dynamic"):
    """
    CORRECTED: Position sizing using Kelly Criterion
    - Uses actual account balance (not hardcoded 100 USDT)
    - Applies Kelly fraction based on proven edge
    """
    
    if capital == "dynamic":
        actual_balance = get_balance()
        if actual_balance <= 0:
            actual_balance = 100.0
    else:
        actual_balance = float(capital)
    
    win_rate, avg_win, avg_loss, profit_factor, expectancy, sharpe = calculate_recent_edge(lookback_trades=50)
    
    if win_rate < 0.33 or expectancy <= 0:
        kelly_f = 0.0
        log_event(f"⚠ [KELLY] No edge detected (WR={win_rate:.1%}). Reducing position to 0.")
    else:
        kelly_f = (win_rate - (1 - win_rate) * (avg_loss / avg_win)) / (avg_win / avg_loss)
        kelly_f = max(0, kelly_f)
        kelly_f_conservative = kelly_f * 0.25
        log_event(f"[KELLY] Full={kelly_f:.3f} | Conservative (25%)={kelly_f_conservative:.3f} | Sharpe={sharpe:.2f}")
        kelly_f = kelly_f_conservative
    
    risk_multiplier = {"baixo": 0.5, "medio": 1.0, "alto": 1.5}.get(risk_level, 1.0)
    adjusted_kelly = kelly_f * risk_multiplier
    
    max_risk_usd = actual_balance * 0.005 if risk_level == "baixo" else \
                   actual_balance * 0.0075 if risk_level == "medio" else \
                   actual_balance * 0.01
    
    position_usd = actual_balance * adjusted_kelly
    position_usd = min(position_usd, max_risk_usd)
    
    qty = position_usd / current_price
    qty = max(0.001, min(qty, actual_balance * 0.05 / current_price))
    
    return round(qty, 3)


def detect_market_regime(df):
    """Detect market regime: trending, ranging, or high_vol"""
    if len(df) < 30:
        return "normal"
    
    try:
        atr_recent = df['atr'].iloc[-1]
        atr_20ma = df['atr'].rolling(20).mean().iloc[-1]
        vol_ratio = atr_recent / atr_20ma if atr_20ma > 0 else 1.0
        
        ema_50 = df['ema_50'].iloc[-1]
        ema_200 = df['ema_200'].iloc[-1]
        trend_strength = abs(ema_50 - ema_200) / ema_200 if ema_200 > 0 else 0
        
        if trend_strength > 0.02 and vol_ratio < 1.2:
            return "trending"
        elif vol_ratio > 1.3:
            return "high_vol"
        else:
            return "ranging"
    except:
        return "normal"


def calculate_tp_sl_dynamic(price, atr, side, market_regime="normal"):
    """CORRECTED: Regime-adaptive TP/SL targeting 1:2+ risk:reward ratio"""
    
    if market_regime == "trending":
        tp_mult = 2.0
        sl_mult = 0.8
        ratio = 2.5
    elif market_regime == "high_vol":
        tp_mult = 1.5
        sl_mult = 0.9
        ratio = 1.67
    else:
        tp_mult = 1.0
        sl_mult = 1.0
        ratio = 1.0
    
    # FIX: Normalize side to title case for comparison
    side_normalized = side.title() if isinstance(side, str) else "Buy"
    
    if side_normalized == "Buy":
        take_profit = round(price + (atr * tp_mult), 1)
        stop_loss = round(price - (atr * sl_mult), 1)
    else:
        take_profit = round(price - (atr * tp_mult), 1)
        stop_loss = round(price + (atr * sl_mult), 1)
    
    log_event(f"[TP/SL] Regime={market_regime} | Ratio={ratio:.2f} | TP±{tp_mult*atr:.0f} | SL±{sl_mult*atr:.0f}")
    
    return take_profit, stop_loss


def salvar_log_csv(data):
    try:
        header = list(data.keys())
        file_exists = Path(LOG_FILE).exists()

        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
    except Exception as e:
        print(f"[ERRO] Falha ao salvar log CSV: {e}")


def calcular_performance_advanced():
    """CORRECTED: Regime-aware weight adaptation with stratified smoothing"""
    global peso_tecnico, peso_sentimento
    
    try:
        if not Path(LOG_FILE).exists():
            return
        
        df_trades = pd.read_csv(LOG_FILE)
        if len(df_trades) < 20:
            return
        
        recent_trades = df_trades.tail(50).copy()
        
        # Stratify by market regime
        try:
            recent_trades['is_trending'] = recent_trades['atr'] > recent_trades['atr'].rolling(20).mean() * 1.1
        except:
            recent_trades['is_trending'] = False
        
        # Global average calculation
        if len(recent_trades) >= 10:
            pnl_tecnico_total = recent_trades[recent_trades['decision_source'] == 'tecnico']['pnl'].sum()
            pnl_sentimento_total = recent_trades[recent_trades['decision_source'] == 'sentimento']['pnl'].sum()
            
            if pnl_tecnico_total + pnl_sentimento_total == 0:
                log_event("[ADAPT] Both technical and sentiment in drawdown. Maintaining weights.")
                return
            
            peso_t_calc = max(0.01, pnl_tecnico_total)
            peso_s_calc = max(0.01, pnl_sentimento_total)
            
            soma = peso_t_calc + peso_s_calc
            alvo_tecnico = peso_t_calc / soma
            alvo_sentimento = peso_s_calc / soma
            
            # Adaptive smoothing based on Sharpe confidence
            try:
                recent_trades_10 = recent_trades.tail(10)
                sharpe_10 = (recent_trades_10['pnl_pct'].mean() / recent_trades_10['pnl_pct'].std()) * np.sqrt(252 / 10)
            except:
                sharpe_10 = 0.0
            
            if abs(sharpe_10) < 0.5:
                taxa_suavizacao = 0.05  # Low confidence → move slow
            elif abs(sharpe_10) > 1.5:
                taxa_suavizacao = 0.20  # High confidence → move fast
            else:
                taxa_suavizacao = 0.10  # Default
            
            peso_tecnico = (peso_tecnico * (1 - taxa_suavizacao)) + (alvo_tecnico * taxa_suavizacao)
            peso_sentimento = (peso_sentimento * (1 - taxa_suavizacao)) + (alvo_sentimento * taxa_suavizacao)
            
            # Softer bounds (0.05-0.95 instead of 0.1-0.9)
            peso_tecnico = np.clip(peso_tecnico, 0.05, 0.95)
            peso_sentimento = 1 - peso_tecnico
            
            log_event(f"[ADAPT-FINAL] Técnico={peso_tecnico:.2f} | Sentimento={peso_sentimento:.2f} | "
                     f"Sharpe(10)={sharpe_10:.2f} | Smoothing={taxa_suavizacao:.2%}")
    
    except Exception as e:
        log_event(f"[ERROR] Advanced adaptation: {e}")


def model_predict_prob_corrected(row, technical_confidence, sinal_tecnico):
    """
    CORRECTED: ML filter that respects technical direction to avoid divergence trades.
    """
    global model, ultima_modificacao_modelo
    
    reload_path = ENSEMBLE_PATH if ENSEMBLE_PATH.exists() else MODEL_PATH
    if reload_path.exists():
        modificacao_atual = os.path.getmtime(reload_path)
        if modificacao_atual > ultima_modificacao_modelo:
            model = joblib.load(reload_path)
            ultima_modificacao_modelo = modificacao_atual
            log_event("🧠 [HOT-RELOAD] ML model updated")
    
    if model is None:
        return None
    
    try:
        if hasattr(model, "num_feature") and model.num_feature() != len(FEATURES):
            log_event(
                f"[ML] Modelo incompatível com features estacionarias "
                f"({model.num_feature()} != {len(FEATURES)}). Retreine ml/train_model.py."
            )
            return None

        df_row = pd.DataFrame([row])
        for f in FEATURES:
            if f not in df_row.columns:
                df_row[f] = 0.0
        
        X = df_row[FEATURES].astype(float).fillna(0)
        
        if isinstance(model, list):
            preds = []
            for sub_model in model:
                if hasattr(sub_model, "predict_proba"):
                    preds.append(float(sub_model.predict_proba(X)[0][1]))
                else:
                    preds.append(float(sub_model.predict(X)[0]))
            prob = float(np.mean(preds)) if preds else 0.5
        elif hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[0][1]
        else:
            prob = model.predict(X)[0]
        
        prob = float(prob)
        if CALIBRATOR_PATH.exists():
            try:
                calibrator = joblib.load(CALIBRATOR_PATH)
                prob = float(calibrator.predict_proba([[prob]])[0][1])
            except Exception as calib_error:
                log_event(f"[ML] Calibrator unavailable, using raw probability: {calib_error}")

        ml_confidence = 0.0
        
        # CORRECTED: Avaliação direcional com base no sinal técnico
        if sinal_tecnico == "buy":
            if prob < 0.50:
                log_event(f"❌ [ML] DIVERGÊNCIA: Sinal Técnico quer BUY, mas ML quer Bearish ({prob:.2%})")
                ml_confidence = 0.0
            elif 0.50 <= prob < 0.60:
                ml_confidence = 0.50
                log_event(f"[ML] Weak bullish ({prob:.2%}), moderate confidence")
            elif 0.60 <= prob < 0.70:
                ml_confidence = 0.75
                log_event(f"[ML] Moderate bullish ({prob:.2%}), high confidence")
            else:
                ml_confidence = 1.0
                log_event(f"[ML] Strong bullish ({prob:.2%}), maximum confidence")
                
        elif sinal_tecnico == "sell":
            if prob > 0.50:
                log_event(f"❌ [ML] DIVERGÊNCIA: Sinal Técnico quer SELL, mas ML quer Bullish ({prob:.2%})")
                ml_confidence = 0.0
            elif 0.40 < prob <= 0.50:
                ml_confidence = 0.50
                log_event(f"[ML] Weak bearish ({(1-prob):.2%}), moderate confidence")
            elif 0.30 < prob <= 0.40:
                ml_confidence = 0.75
                log_event(f"[ML] Moderate bearish ({(1-prob):.2%}), high confidence")
            else:
                ml_confidence = 1.0
                log_event(f"[ML] Strong bearish ({(1-prob):.2%}), maximum confidence")
                
        else:
            ml_confidence = 0.0
        
        # Blend with technical
        final_confidence = (0.7 * (technical_confidence / 100)) + (0.3 * ml_confidence)
        final_confidence_pct = final_confidence * 100
        
        if final_confidence_pct >= 65:
            log_event(f"✅ [DECISION] ENTER | Blended confidence: {final_confidence_pct:.1f}%")
            return prob
        else:
            log_event(f"❌ [DECISION] REJECT | Blended confidence: {final_confidence_pct:.1f}% < 65%")
            return None
        
    except Exception as e:
        log_event(f"[ERROR] ML prediction: {e}")
        return None


def atualizar_cache_candle_ws(candle):
    """Append a confirmed WebSocket kline to the in-memory OHLCV cache."""
    global historico_candles_ws

    try:
        timestamp = normalize_timestamp_ms_value(candle.get("start") or candle.get("timestamp") or candle.get("time"))
        if timestamp is None:
            raise ValueError("timestamp ausente")
        row = {
            "timestamp": timestamp,
            "open": float(candle.get("open")),
            "high": float(candle.get("high")),
            "low": float(candle.get("low")),
            "close": float(candle.get("close")),
            "volume": float(candle.get("volume")),
        }
    except Exception as e:
        log_event(f"[WS] Candle invalido recebido: {e}")
        return

    with candle_cache_lock:
        historico_candles_ws = pd.concat([historico_candles_ws, pd.DataFrame([row])], ignore_index=True)
        historico_candles_ws["timestamp"] = normalize_timestamp_ms_series(historico_candles_ws["timestamp"])
        historico_candles_ws = (
            historico_candles_ws
            .dropna(subset=["timestamp"])
            .drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .tail(500)
            .reset_index(drop=True)
        )


def obter_candles_para_analise():
    """Prefer WebSocket-fed candles and fall back to REST only while warming up."""
    with candle_cache_lock:
        df_ws = historico_candles_ws.copy()

    if "timestamp" in df_ws.columns and not df_ws.empty:
        df_ws["timestamp"] = normalize_timestamp_ms_series(df_ws["timestamp"])
        df_ws = df_ws.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if len(df_ws) >= 320:
        return df_ws

    df_rest = get_ohlcv(SYMBOL, interval="5", limit=1000)
    if not df_rest.empty:
        df_rest["timestamp"] = normalize_timestamp_ms_series(df_rest["timestamp"])
        df_rest = df_rest.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        with candle_cache_lock:
            globals()["historico_candles_ws"] = df_rest.tail(500).reset_index(drop=True)
        log_event("[DATA] Cache WebSocket aquecida via REST inicial.")
    return df_rest


def anexar_metricas_derivativos(df, metrics):
    df = df.copy()
    keys = ["funding_rate", "predicted_funding_rate", "premium_index", "premium_basis_pct"]
    for key in keys:
        if key not in df.columns:
            df[key] = np.nan
    if not df.empty:
        last_idx = df.index[-1]
        for key in keys:
            df.at[last_idx, key] = float(metrics.get(key, 0.0) or 0.0)
    return df


def persist_latest_market_snapshot(df):
    """Persist latest exogenous metrics in the candle cache for the next cycle."""
    global historico_candles_ws

    if df.empty or "timestamp" not in df.columns:
        return

    metric_cols = [
        "funding_rate", "predicted_funding_rate", "premium_index", "premium_basis_pct",
        "cvd", "cvd_ratio", "oi", "oi_change_pct",
        "liquidation_imbalance", "liquidation_notional", "liquidation_reversal_signal",
        "liquidation_cluster_density", "liquidation_cluster_side",
        "spot_cvd_ratio", "perp_cvd_ratio", "spot_perp_cvd_divergence",
        "sentiment_score", "risk_level_encoded",
    ]
    last = df.iloc[-1]
    timestamp = normalize_timestamp_ms_value(last.get("timestamp"))
    if timestamp is None:
        return

    with candle_cache_lock:
        if historico_candles_ws.empty or "timestamp" not in historico_candles_ws.columns:
            return
        historico_candles_ws["timestamp"] = normalize_timestamp_ms_series(historico_candles_ws["timestamp"])
        mask = historico_candles_ws["timestamp"] == timestamp
        if not mask.any():
            return
        idx = historico_candles_ws.index[mask][-1]
        for col in metric_cols:
            if col in df.columns:
                if col not in historico_candles_ws.columns:
                    historico_candles_ws[col] = np.nan
                historico_candles_ws.at[idx, col] = last.get(col)


def obter_metricas_order_flow(df):
    with market_flow_lock:
        trades_ws = list(trade_tape_ws)
        liquidation_events = list(liquidation_events_ws)

    perp_trades = trades_ws or get_public_trades(SYMBOL, limit=200, category="linear")
    spot_trades = get_public_trades(SYMBOL, limit=200, category="spot")
    cvd_metrics = calculate_cvd_from_trades(perp_trades)
    if spot_trades or perp_trades:
        cvd_metrics.update(calculate_spot_perp_cvd_divergence(spot_trades, perp_trades))
    if not perp_trades:
        cvd_metrics = estimate_cvd_from_candles(df)

    oi_metrics = get_open_interest_metrics(SYMBOL, interval_time="5min", limit=2)
    liquidation_metrics = calculate_liquidation_metrics(liquidation_events)

    log_event(
        "[FLOW] CVD={:.4f} ({:.2%}) | OIΔ={:.2%} | LiqImb={:.2%}".format(
            cvd_metrics.get("cvd", 0.0),
            cvd_metrics.get("cvd_ratio", 0.0),
            oi_metrics.get("oi_change_pct", 0.0),
            liquidation_metrics.get("liquidation_imbalance", 0.0),
        )
    )
    return cvd_metrics, oi_metrics, liquidation_metrics


def abrir_ordem(df=None):
    global ultima_ordem, bloqueio_ate

    if not enforce_daily_drawdown_circuit_breaker():
        return

    if datetime.now() < bloqueio_ate:
        log_event(f"⏳ Bot em modo Circuit Breaker. Operações suspensas até {bloqueio_ate.strftime('%H:%M:%S')}.")
        return

    df = df.copy() if df is not None else obter_candles_para_analise()
    if df.empty:
        return

    if "timestamp" in df.columns:
        df["timestamp"] = normalize_timestamp_ms_series(df["timestamp"])
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    df = calculate_indicators(df)
    derivatives_metrics = get_derivatives_metrics(SYMBOL)
    df = anexar_metricas_derivativos(df, derivatives_metrics)
    cvd_metrics, oi_metrics, liquidation_metrics = obter_metricas_order_flow(df)
    df = add_order_flow_features(df, cvd_metrics, oi_metrics, liquidation_metrics)
    log_event(
        "[DERIV] Funding={:.4%} | PredFunding={:.4%} | Premium={:.4%}".format(
            derivatives_metrics.get("funding_rate", 0.0) or 0.0,
            derivatives_metrics.get("predicted_funding_rate", 0.0) or 0.0,
            derivatives_metrics.get("premium_basis_pct", 0.0) or 0.0,
        )
    )

    log_event(f"📊 Preço de fechamento real (DataFrame): {df['close'].iloc[-1]}")

    sentimento_str, sent_score = get_news_sentiment("BTC")
    if sentimento_str == "bullish":
        confiança_sentimento = 100
    elif sentimento_str == "bearish":
        confiança_sentimento = 0
    else:
        confiança_sentimento = 50

    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = np.nan
    df.at[df.index[-1], "sentiment_score"] = sent_score

    df = prepare_features(df)
    market_regime = detect_market_regime(df)

    log_event(f"Sentimento: {sentimento_str} ({confiança_sentimento}%)")

    trade_decision = generate_trade_signal(df, derivatives_metrics=derivatives_metrics)
    sinal_tecnico = trade_decision.get("signal")
    confiança_tecnica = trade_decision.get("confidence", 50.0)
    if market_regime == "ranging" and sinal_tecnico in ["buy", "sell"]:
        log_event("[REGIME] Momentum signal blocked in ranging regime.")
        return

    confiança_final = (peso_tecnico * confiança_tecnica) + (peso_sentimento * confiança_sentimento)

    if confiança_final >= 80:
        risk_level = "baixo"
        risk_level_encoded = 0
    elif confiança_final >= 60:
        risk_level = "medio"
        risk_level_encoded = 1
    else:
        risk_level = "alto"
        risk_level_encoded = 2

    log_event(f"Sinal técnico: {sinal_tecnico} | Conf. técnica: {confiança_tecnica}% | "
              f"Conf. final: {confiança_final:.1f}% | Risco: {risk_level}")

    df.at[df.index[-1], "risk_level_encoded"] = risk_level_encoded
    persist_latest_market_snapshot(df)

    df_ml = apply_strict_feature_lag(df, FEATURES, periods=1)
    prob_ml = model_predict_prob_corrected(df_ml.iloc[-1].to_dict(), confiança_tecnica, sinal_tecnico)
    if prob_ml is None:
        log_event("⚠ ML filtering rejected entry - insufficient blended confidence.")
        return

    ultimo = df.iloc[-1]
    print(f"[DEBUG] ML prob: {prob_ml:.2f} | Sentimento: {sentimento_str} ({sent_score:.2f})")
    print(f"[DEBUG] RSI: {ultimo['rsi']:.2f} | MACD: {ultimo['macd']:.2f} | "
          f"Signal: {ultimo['macd_signal']:.2f} | EMA20: {ultimo['ema_20']:.2f} | EMA50: {ultimo['ema_50']:.2f}")
    print(f"[DEBUG] Decisão: {'Abrir trade' if sinal_tecnico in ['buy', 'sell'] else 'Ignorar sinal'}")

    if sinal_tecnico not in ["buy", "sell"]:
        return

    posicoes = get_all_positions(SYMBOL) or []
    
    # ADICIONADO: Leitura da ultima_ordem protegida por Thread Lock para evitar erros
    with ordem_lock:
        cooldown_ativo = (ultima_ordem.get("side") == sinal_tecnico and
                          datetime.now() - ultima_ordem.get("hora", datetime.min) < timedelta(seconds=10) and
                          len(posicoes) == 0)
        
    if cooldown_ativo:
        print("⚠ Ordem recente no mesmo sentido (cooldown). Ignorando.")
        return

    for pos in posicoes:
        if pos.get("side", "").lower() == sinal_tecnico:
            print(f"⚠ Já existe posição {sinal_tecnico.upper()} aberta. Ordem ignorada.")
            return

    price = get_last_price(SYMBOL)
    if price is None:
        return

    qty = calculate_order_qty_kelly_criterion(SYMBOL, risk_level, price)
    
    # Arredonda a QTY para 3 casas decimais (Step Size do BTC na Bybit)
    qty = round(qty, 3) 

    atr_atual = df.iloc[-1]['atr']
    
    # CORRECTED: Dynamic TP/SL with regime detection
    side = "Buy" if sinal_tecnico == "buy" else "Sell"
    exits = calculate_atr_exit_prices(price, atr_atual, side, market_regime)
    take_profit = exits["take_profit"]
    stop_loss = exits["stop_loss"]
    
    trailing_stop = round((atr_atual * 0.4), 1)
    
    # Set side based on signal
    maker_price = get_maker_price(SYMBOL, side)
    if maker_price is None:
        log_event("⚠ Ordem maker cancelada: spread acima do limite ou book indisponivel.")
        return

    if not slippage_within_threshold(price, maker_price, side, max_slippage_pct=0.0008):
        log_event(f"⚠ Ordem maker cancelada por slippage esperado: ref={price} maker={maker_price}")
        return

    log_event(f"Abrindo {side} | Preço Desejado: {price} | TP: {take_profit} | SL: {stop_loss} | TS: {trailing_stop}")
    log_event(
        f"[MAKER] Limit={maker_price} | ATR SLx={exits['sl_atr_multiple']} TPx={exits['tp_atr_multiple']}"
    )
    order_result = place_maker_order(
        SYMBOL, side, str(qty),
        str(take_profit),
        str(stop_loss),
        str(trailing_stop),
        price=maker_price,
    )

    ok = False
    if isinstance(order_result, dict):
        ok = (order_result.get("retCode") == 0) or (order_result.get("success") is True) or bool(order_result.get("result"))
    else:
        ok = bool(order_result)

    if not ok:
        log_event(f"❌ Falha ao abrir ordem {side} em {SYMBOL}: {order_result}")
        return

    order_id = None
    if isinstance(order_result, dict):
        order_id = order_result.get("result", {}).get("orderId")

    # ----------------------------------------------------------------------------------
    # ADICIONADO: PROTEÇÃO CONTRA SLIPPAGE (Captura do Preço de Execução Real)
    # ----------------------------------------------------------------------------------
    time.sleep(1) # Dá 1 segundo para a corretora preencher a Market Order
    posicoes_pos_ordem = get_all_positions(SYMBOL) or []
    preco_real_execucao = float(price) # fallback caso a API demore
    ordem_preenchida = False
    
    for p in posicoes_pos_ordem:
        if p.get("side", "").lower() == side.lower():
            avg_price = float(p.get("avgPrice", 0))
            if avg_price > 0:
                preco_real_execucao = avg_price
                ordem_preenchida = True
                log_event(f"✅ Slippage Calculado. Executado a: {preco_real_execucao} (Solicitado: {price})")
                break
    # ----------------------------------------------------------------------------------

    if not ordem_preenchida:
        if order_id:
            cancel_order(SYMBOL, order_id=order_id)
        log_event("⚠ Ordem maker enviada, mas ainda nao preenchida. Aguardando proximo ciclo.")
        return

    if not slippage_within_threshold(price, preco_real_execucao, side, max_slippage_pct=0.0008):
        if order_id:
            cancel_order(SYMBOL, order_id=order_id)
        log_event(f"⚠ Slippage real acima do limite: ref={price} exec={preco_real_execucao}.")
        return

    trade_id = str(uuid.uuid4())
    decision_doc = {
        "trade_id": trade_id,
        "timestamp": datetime.now(),
        "symbol": SYMBOL,
        "side": side,
        "entry_price": float(preco_real_execucao), # EDITADO: Salva o preço que a Bybit cobrou de verdade
        "qty": float(qty),
        "risk_level": risk_level,
        "decision_source": ("tecnico" if peso_tecnico > peso_sentimento
                            else "sentimento" if peso_sentimento > peso_tecnico
                            else "misto"),
        "confidence": {
            "technical": float(confiança_tecnica),
            "sentiment": float(confiança_sentimento),
            "final": float(confiança_final),
        },
        "ml_probability": float(prob_ml) if prob_ml is not None else None,
        "targets": {
            "take_profit": float(take_profit),
            "stop_loss": float(stop_loss),
            "trailing_stop": float(trailing_stop),
        },
        "features": {k: float(ultimo[k]) for k in FEATURES if k in ultimo}
    }
    
    try:
        log_signal_decision(decision_doc)
    except Exception as db_err:
        log_event(f"⚠ Erro ao salvar decisão no DB, mas trade continuará: {db_err}")

    # ADICIONADO: O Thread Lock na hora de salvar o dicionário da sessão
    with ordem_lock:
        ultima_ordem = {
            "trade_id": trade_id,
            "side": sinal_tecnico,
            "hora": datetime.now(),
            "origem": decision_doc["decision_source"],
            "risk_level": risk_level,
            "ml_probability": prob_ml,
            "hour": int(df.iloc[-1]["hour"]),
            "minute": int(df.iloc[-1]["minute"]),
            "ema_20": df.iloc[-1]["ema_20"],
            "ema_50": df.iloc[-1]["ema_50"],
            "ema_200": df.iloc[-1]["ema_200"],
            "rsi": df.iloc[-1]["rsi"],
            "macd": df.iloc[-1]["macd"],
            "macd_signal": df.iloc[-1]["macd_signal"],
            "macd_hist": df.iloc[-1]["macd_hist"],
            "bb_width": df.iloc[-1]["bb_width"],
            "atr": df.iloc[-1]["atr"],
            "volume": df.iloc[-1]["volume"],
            "volume_ma": df.iloc[-1]["volume_ma"],
            "sentiment_str": sentimento_str,
            "sentiment_score": sent_score,
            "take_profit": float(take_profit),
            "stop_loss": float(stop_loss),
            "trailing_stop": float(trailing_stop),
            "entry_price": float(preco_real_execucao), # EDITADO: Repassa o preço real
            "qty": float(qty),
        }

def fechar_ordem(side, entry_price, size, current_price):
    close_position(SYMBOL, side)

    taxa_corretora = 0.0005
    custo_taxas = (entry_price * size * taxa_corretora) + (current_price * size * taxa_corretora)
    
    pnl_bruto = (current_price - entry_price) * size if side == "Buy" else (entry_price - current_price) * size
    pnl = pnl_bruto - custo_taxas
    
    pnl_pct = ((current_price / entry_price) - 1) * 100 if side == "Buy" else ((entry_price / current_price) - 1) * 100

    # Adicionado o Lock aqui também apenas por garantia, já que lê a variável global
    with ordem_lock:
        ordem_memoria = ultima_ordem.copy()

    trade_data = {
        "timestamp": datetime.now(),
        "trade_id": ordem_memoria.get("trade_id"),
        "symbol": SYMBOL,
        "side": side,
        "entry_price": float(entry_price),
        "exit_price": float(current_price),
        "qty": float(size),
        "pnl": round(float(pnl), 2),
        "pnl_pct": round(float(pnl_pct), 3),
        "decision_source": ordem_memoria.get("origem", "misto"),
        "risk_level": ordem_memoria.get("risk_level", "desconhecido"),
        "take_profit": ordem_memoria.get("take_profit"),
        "stop_loss": ordem_memoria.get("stop_loss"),
        "trailing_stop": ordem_memoria.get("trailing_stop"),
        "sentiment_str": ordem_memoria.get("sentiment_str"),
        "sentiment_score": ordem_memoria.get("sentiment_score"),
        "ml_probability": ordem_memoria.get("ml_probability"),
        "ema_20": ordem_memoria.get("ema_20"),
        "ema_50": ordem_memoria.get("ema_50"),
        "ema_200": ordem_memoria.get("ema_200"),
        "rsi": ordem_memoria.get("rsi"),
        "macd": ordem_memoria.get("macd"),
        "macd_signal": ordem_memoria.get("macd_signal"),
        "macd_hist": ordem_memoria.get("macd_hist"),
        "bb_width": ordem_memoria.get("bb_width"),
        "atr": ordem_memoria.get("atr"),
        "volume": ordem_memoria.get("volume"),
        "volume_ma": ordem_memoria.get("volume_ma"),
        "hour": ordem_memoria.get("hour"),
        "minute": ordem_memoria.get("minute")
    }

    salvar_log_csv(trade_data)
    log_event(f"{side} fechado - PnL: {pnl:.2f} ({pnl_pct:.3f}%) | Origem: {trade_data['decision_source']}")
    
    try:
        log_trade(trade_data)
        trade_id = trade_data.get("trade_id")
        if trade_id:
            label = 1 if pnl > 0 else 0
            update_signal_outcome(trade_id, {
                "exit_price": float(current_price),
                "pnl": float(round(pnl, 2)),
                "pnl_pct": float(round(pnl_pct, 3)),
                "label": int(label),
                "status": "closed",
                "closed_at": datetime.utcnow()
            })
    except Exception as e:
        log_event(f"⚠ Falha ao salvar no banco em fechar_ordem: {e}")

    with ordem_lock:
        ultima_ordem["side"] = None
        ultima_ordem["hora"] = datetime.min

def monitorar_posicoes():
    posicoes = get_all_positions(SYMBOL)
    if not posicoes:
        return

    # ADICIONADO: Criação de um "clone" congelado no tempo para o while loop ler com segurança
    with ordem_lock:
        ordem_atual = ultima_ordem.copy()

    for pos in posicoes:
        entry_price = float(pos.get("avgPrice", 0))
        side = pos.get("side")
        size = float(pos.get("size", 0))
        current_price = get_last_price(SYMBOL)

        if not current_price:
            continue

        print(f"[MONITOR] {side} {size} @ {entry_price} | Preço atual: {current_price}")

        fechar = False
        tp_real = ordem_atual.get("take_profit", entry_price * 1.02)
        sl_real = ordem_atual.get("stop_loss", entry_price * 0.98)

        if side == "Buy":
            if current_price >= tp_real or current_price <= sl_real:
                fechar = True
        elif side == "Sell":
            if current_price <= tp_real or current_price >= sl_real:
                fechar = True
                
        if fechar:
                close_position(SYMBOL, side)

                taxa_corretora = 0.0005
                custo_taxas = (entry_price * size * taxa_corretora) + (current_price * size * taxa_corretora)

                pnl_bruto = (current_price - entry_price) * size if side == "Buy" else (entry_price - current_price) * size
                pnl = pnl_bruto - custo_taxas
                
                pnl_pct = ((current_price / entry_price) - 1) * 100 if side == "Buy" else ((entry_price / current_price) - 1) * 100

                trade_data = {
                    "timestamp": datetime.now(),
                    "trade_id": ordem_atual.get("trade_id"), # EDITADO: Puxando da variável travada
                    "symbol": SYMBOL,
                    "side": side,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "qty": size,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 3),
                    "decision_source": ordem_atual.get("origem", "misto"),
                    "risk_level": ordem_atual.get("risk_level", "desconhecido"),
                    "take_profit": ordem_atual.get("take_profit"),
                    "stop_loss": ordem_atual.get("stop_loss"),
                    "sentiment_str": ordem_atual.get("sentiment_str"),
                    "sentiment_score": ordem_atual.get("sentiment_score"),
                    "ml_probability": ordem_atual.get("ml_probability"),
                    "ema_20": ordem_atual.get("ema_20"),
                    "ema_50": ordem_atual.get("ema_50"),
                    "ema_200": ordem_atual.get("ema_200"),
                    "rsi": ordem_atual.get("rsi"),
                    "macd": ordem_atual.get("macd"),
                    "macd_signal": ordem_atual.get("macd_signal"),
                    "macd_hist": ordem_atual.get("macd_hist"),
                    "bb_width": ordem_atual.get("bb_width"),
                    "atr": ordem_atual.get("atr"),
                    "volume": ordem_atual.get("volume"),
                    "volume_ma": ordem_atual.get("volume_ma"),
                    "hour": ordem_atual.get("hour"),
                    "minute": ordem_atual.get("minute")
                }

                salvar_log_csv(trade_data)
                log_event(f"{side} fechado - PnL: {pnl:.2f} ({pnl_pct:.3f}%) | Origem: {trade_data['decision_source']}")
                
                # ADICIONADO: Proteção (Try...Except) para queda do banco de dados na hora de salvar lucros
                try:
                    log_trade(trade_data)

                    trade_id = trade_data.get("trade_id")
                    if trade_id:
                        label = 1 if pnl > 0 else 0
                        update_signal_outcome(trade_id, {
                            "exit_price": float(current_price),
                            "pnl": float(round(pnl, 2)),
                            "pnl_pct": float(round(pnl_pct, 3)),
                            "label": int(label)
                        })
                except Exception as erro_banco:
                    log_event(f"⚠ O trade foi fechado e o lucro garantido, mas houve falha ao salvar no banco Mongo: {erro_banco}")

                # ADICIONADO: Limpeza crucial que havia sido esquecida neste bloco
                with ordem_lock:
                    ultima_ordem["side"] = None
                    ultima_ordem["hora"] = datetime.min


# =======================================================
if __name__ == "__main__":
    from pybit.unified_trading import WebSocket
    import threading
    
    log_event("🚀 Bot iniciado. Configurando WebSockets...")

    def handle_kline(message):
        """Callback acionado pela Bybit"""
        try:
            # 1. Escudo Global: Garante que o Python veja as variáveis da Thread Principal
            global ultimo_candle_recebido, in_position, session_metrics, model_weights, regime_history
            
            data = message.get("data", [])
            
            # Se a corretora mandar um pacote vazio, ignora e protege o código
            if not data: 
                return
                
            candle = data[0]
            
            # 2. Quando o candle fechar oficialmente (confirm = True)
            if candle.get("confirm"):
                atualizar_cache_candle_ws(candle)
                
                # Reseta o relógio do Watchdog para provar que a conexão está viva
                ultimo_candle_recebido = datetime.now() 
                
                log_event("🕯️ Candle fechado! Iniciando extração de dados da Corretora e análise de ML...")
                
                # 3. Processamento em Sub-Thread para o WebSocket não engasgar
                def executar_trabalho_pesado():
                    try:
                        calcular_performance_advanced()
                        abrir_ordem(obter_candles_para_analise())
                    except Exception as e:
                        log_event(f"❌ Erro durante a análise pós-candle: {e}")

                import threading
                thread_analise = threading.Thread(target=executar_trabalho_pesado)
                thread_analise.start()
                
        except Exception as e:
            # 4. Escudo Final: Se QUALQUER coisa der errado, loga o erro mas NÃO mata a conexão
            log_event(f"🚨 [CRÍTICO] Erro interno ignorado no WebSocket (A thread foi salva!): {e}")

    def handle_public_trade(message):
        data = message.get("data", [])
        if not isinstance(data, list):
            data = [data]
        with market_flow_lock:
            trade_tape_ws.extend(data)
            del trade_tape_ws[:-500]

    def handle_liquidation(message):
        data = message.get("data", [])
        if not isinstance(data, list):
            data = [data]
        normalized = []
        for event in data:
            normalized.append({
                "side": event.get("side") or event.get("S"),
                "price": event.get("price") or event.get("p"),
                "qty": event.get("size") or event.get("qty") or event.get("v"),
            })
        with market_flow_lock:
            liquidation_events_ws.extend(normalized)
            del liquidation_events_ws[:-500]

    ws = WebSocket(
            testnet=TESTNET_MODE, # Mude para True se for operar em papel
            channel_type=WEBSOCKET_CHANNEL_TYPE,
        )

    ws.kline_stream(
        interval=WS_INTERVAL,
        symbol=SYMBOL,
        callback=handle_kline
    )

    log_event("📡 WebSocket conectado! Aguardando o fechamento do próximo candle.")

    if hasattr(ws, "public_trade_stream"):
        try:
            ws.public_trade_stream(symbol=SYMBOL, callback=handle_public_trade)
            log_event("[WS] Inscrito no Public Trade Stream com sucesso.")
        except Exception as e:
            if "already subscribed" in str(e).lower():
                pass # Ignora silenciosamente se já estiver inscrito na memória
            else:
                log_event(f"[WS] Erro ao assinar trades: {e}")
    else:
        log_event("[WS] public_trade_stream indisponivel; CVD usara fallback REST.")

    # 2. Inscrição no fluxo de Liquidações (Priorizando V5 - all_liquidation_stream)
    if hasattr(ws, "all_liquidation_stream"):
        try:
            ws.all_liquidation_stream(symbol=SYMBOL, callback=handle_liquidation)
            log_event("[WS] Inscrito no Liquidation Stream (V5) com sucesso.")
        except Exception as e:
            if "already subscribed" in str(e).lower():
                pass # Ignora silenciosamente se já estiver inscrito na memória
            else:
                log_event(f"[WS] Erro ao assinar liquidações: {e}")
    else:
        log_event("[WS] all_liquidation_stream indisponivel nesta versão do pybit.")

    while True:
        try:
            # ADICIONADO: Cão de Guarda (Watchdog) para reinicialização em caso de queda silenciosa
            agora = datetime.now()
            if (agora - ultimo_candle_recebido).total_seconds() > WATCHDOG_TIMEOUT_SECONDS:
                log_event("💀 [WATCHDOG] O WebSocket da Bybit parou de enviar dados (Estado Zumbi). Reiniciando conexão...")
                
                try:
                    if hasattr(ws, "ws"):
                        ws.ws.close()
                    elif hasattr(ws, "close"):
                        ws.close()
                except Exception as close_error:
                    log_event(f"⚠ Falha ao fechar WebSocket antigo: {close_error}")
                
                time.sleep(15)
                
                try:
                    ws = WebSocket(testnet=TESTNET_MODE, channel_type=WEBSOCKET_CHANNEL_TYPE)
                    ws.kline_stream(interval=WS_INTERVAL, symbol=SYMBOL, callback=handle_kline)

                    if hasattr(ws, "public_trade_stream"):
                        try:
                            ws.public_trade_stream(symbol=SYMBOL, callback=handle_public_trade)
                        except Exception as e:
                            log_event(f"⚠ Não foi possível reinscrever public_trade_stream: {e}")

                    if hasattr(ws, "all_liquidation_stream"):
                        try:
                            ws.all_liquidation_stream(symbol=SYMBOL, callback=handle_liquidation)
                        except Exception as e:
                            log_event(f"⚠ Não foi possível reinscrever all_liquidation_stream: {e}")

                    log_event("✅ WatchDog reconectou o WebSocket com sucesso.")
                except Exception as reconnect_error:
                    log_event(f"❌ Falha ao reconectar WebSocket no WatchDog: {reconnect_error}")

                ultimo_candle_recebido = datetime.now() # Reseta o tempo após reiniciar

            # Monitora as ordens abertas respeitando o rate limit da Bybit (15s)
            monitorar_posicoes()
            
        except Exception as e:
            log_event(f"❌ Erro ao monitorar posições: {e}")
            
        time.sleep(LOOP_INTERVAL)
