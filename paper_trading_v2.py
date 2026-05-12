"""
PAPER TRADING COMPARISON SCRIPT
Simulates V2 decisions and compares with V1 (live) decisions
Outputs: v1_vs_v2_comparison.csv

Usage:
  Terminal 1: python main.py                    # V1 live bot
  Terminal 2: python paper_trading_v2.py        # V2 simulator
  
  Compare results after 48+ hours
"""

import pandas as pd
import numpy as np
import os
import time
from pathlib import Path
from datetime import datetime, timedelta

# Import required modules (same as main.py)
from utils.ohlcv import get_ohlcv
from strategies.strategy import generate_trade_signal
from sentiment.sentiment_analysis import get_news_sentiment
from trading.bybit_api import get_last_price, get_balance
from trading.logger import log_event
from utils.technical_indicators import calculate_indicators

SYMBOL = "BTCUSDT"
COMPARISON_LOG = "v1_vs_v2_comparison.csv"


def simulate_v2_decision(df, last_entry=None):
    """
    Simulate what V2 (corrected bot) would decide on this candle
    Returns: decision dict with entry signal, qty, confidence, etc.
    """
    
    if df.empty:
        return None
    
    try:
        # Get technical signal (same as V1)
        df = calculate_indicators(df)
        trade_decision = generate_trade_signal(df)
        sinal_tecnico = trade_decision.get("signal")
        confiança_tecnica = trade_decision.get("confidence", 50.0)
        
        if sinal_tecnico not in ["buy", "sell"]:
            return None
        
        # Get sentiment (same as V1)
        sentimento_str, sent_score = get_news_sentiment("BTC")
        if sentimento_str == "bullish":
            confiança_sentimento = 100
        elif sentimento_str == "bearish":
            confiança_sentimento = 0
        else:
            confiança_sentimento = 50
        
        # V2 corrections applied:
        # 1. Calculate edge-based position sizing (simplified)
        price = get_last_price(SYMBOL)
        if price is None:
            return None
        
        balance = get_balance()
        if balance <= 0:
            balance = 100.0
        
        # Simplified Kelly (would use full calculation in real scenario)
        qty_v2 = (balance * 0.001) / price  # 0.1% risk per trade
        qty_v2 = round(qty_v2, 3)
        
        # 2. Calculate regime-adaptive TP/SL (simplified)
        atr = df['atr'].iloc[-1]
        ema_50 = df['ema_50'].iloc[-1]
        ema_200 = df['ema_200'].iloc[-1]
        trend_strength = abs(ema_50 - ema_200) / ema_200 if ema_200 > 0 else 0
        
        if trend_strength > 0.02:
            regime = "trending"
            tp_mult, sl_mult = 2.0, 0.8
        else:
            regime = "ranging"
            tp_mult, sl_mult = 1.0, 1.0
        
        if sinal_tecnico == "buy":
            side = "Buy"
            take_profit = round(price + (atr * tp_mult), 1)
            stop_loss = round(price - (atr * sl_mult), 1)
        else:
            side = "Sell"
            take_profit = round(price - (atr * tp_mult), 1)
            stop_loss = round(price + (atr * sl_mult), 1)
        
        # 3. ML filter (confidence threshold 65%)
        final_confidence = (0.7 * confiança_tecnica / 100) + (0.3 * confiança_sentimento / 100) * 100
        
        if final_confidence < 65:
            return None  # Rejected
        
        return {
            "timestamp": datetime.now(),
            "signal": sinal_tecnico,
            "side": side,
            "confidence": final_confidence,
            "qty": qty_v2,
            "entry_price": price,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "regime": regime,
            "source": "ml_enhanced",
            "technical_confidence": confiança_tecnica,
            "sentiment_confidence": confiança_sentimento
        }
        
    except Exception as e:
        log_event(f"[ERROR] V2 simulation: {e}")
        return None


def log_comparison(v1_decision, v2_decision):
    """
    Log comparison between V1 and V2 decisions
    """
    
    comparison = {
        "timestamp": datetime.now(),
        
        # V1 data
        "v1_signal": v1_decision.get("signal") if v1_decision else None,
        "v1_qty": v1_decision.get("qty") if v1_decision else 0,
        "v1_confidence": v1_decision.get("confidence") if v1_decision else 0,
        "v1_tp": v1_decision.get("take_profit") if v1_decision else None,
        "v1_sl": v1_decision.get("stop_loss") if v1_decision else None,
        
        # V2 data
        "v2_signal": v2_decision.get("signal") if v2_decision else None,
        "v2_qty": v2_decision.get("qty") if v2_decision else 0,
        "v2_confidence": v2_decision.get("confidence") if v2_decision else 0,
        "v2_tp": v2_decision.get("take_profit") if v2_decision else None,
        "v2_sl": v2_decision.get("stop_loss") if v2_decision else None,
        "v2_regime": v2_decision.get("regime") if v2_decision else None,
        
        # Comparison
        "signal_agreement": 1 if (v1_decision and v2_decision and 
                                   v1_decision.get("signal") == v2_decision.get("signal")) else 0,
        "qty_ratio": (v2_decision.get("qty", 0) / max(v1_decision.get("qty", 0.001), 0.001)) 
                     if v1_decision else 0,
        "confidence_delta": (v2_decision.get("confidence", 0) - v1_decision.get("confidence", 0))
                           if v1_decision else 0,
        "v1_entered": 1 if v1_decision else 0,
        "v2_entered": 1 if v2_decision else 0,
    }
    
    df = pd.DataFrame([comparison])
    
    file_exists = Path(COMPARISON_LOG).exists()
    df.to_csv(COMPARISON_LOG, mode='a', header=file_exists, index=False)
    
    return comparison


def print_comparison_summary():
    """Print summary statistics of V1 vs V2 from CSV log"""
    
    if not Path(COMPARISON_LOG).exists():
        print("No comparison data yet. Run for 48+ hours to generate sample.")
        return
    
    try:
        df = pd.read_csv(COMPARISON_LOG)
        
        if len(df) < 10:
            print(f"Insufficient data ({len(df)} entries). Need 50+ for meaningful comparison.")
            return
        
        print("\n" + "=" * 70)
        print("V1 vs V2 COMPARISON SUMMARY")
        print("=" * 70)
        print(f"Total candles analyzed: {len(df)}")
        print(f"Sample period: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        
        print(f"\nEntry Frequency:")
        print(f"  V1 entries: {df['v1_entered'].sum()} ({df['v1_entered'].mean()*100:.1f}%)")
        print(f"  V2 entries: {df['v2_entered'].sum()} ({df['v2_entered'].mean()*100:.1f}%)")
        
        if df['v1_entered'].sum() > 0:
            v1_entered = df[df['v1_entered'] == 1]
            print(f"\nWhen V1 Entered (n={len(v1_entered)}):")
            print(f"  V2 agreed: {v1_entered['signal_agreement'].sum()} ({v1_entered['signal_agreement'].mean()*100:.1f}%)")
            print(f"  Avg V2 signal quality: {v1_entered[v1_entered['v2_signal'].notna()]['v2_confidence'].mean():.1f}%")
            print(f"  V2 qty ratio: {v1_entered[v1_entered['v2_qty'] > 0]['qty_ratio'].mean():.2f}x")
        
        if df['v2_entered'].sum() > 0:
            v2_entered = df[df['v2_entered'] == 1]
            print(f"\nWhen V2 Entered (n={len(v2_entered)}):")
            print(f"  V1 missed: {len(v2_entered[v2_entered['v1_entered'] == 0])} ({len(v2_entered[v2_entered['v1_entered'] == 0])/len(v2_entered)*100:.1f}%)")
            print(f"  Avg V2 confidence: {v2_entered['v2_confidence'].mean():.1f}%")
            
            high_conf_v2 = len(v2_entered[v2_entered['v2_confidence'] >= 70])
            print(f"  High confidence entries (>=70%): {high_conf_v2}/{len(v2_entered)} ({high_conf_v2/len(v2_entered)*100:.1f}%)")
        
        print(f"\nAverage Confidence:")
        print(f"  V1: {df['v1_confidence'].mean():.1f}%")
        print(f"  V2: {df['v2_confidence'].mean():.1f}%")
        print(f"  Delta (V2-V1): {(df['v2_confidence'].mean() - df['v1_confidence'].mean()):+.1f}%")
        
        print(f"\nPosition Sizing Comparison:")
        ratios = df[df['qty_ratio'] > 0]['qty_ratio']
        if len(ratios) > 0:
            print(f"  V2/V1 size ratio: {ratios.mean():.2f}x (median: {ratios.median():.2f}x)")
        
        print("\n" + "=" * 70)
        print("INTERPRETATION:")
        print("• V2 should enter fewer trades but with higher quality")
        print("• V2 confidence should be higher when entering")
        print("• V2 should avoid low-confidence entries that V1 takes")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"Error reading comparison log: {e}")


def main():
    """Main paper trading loop"""
    
    print("\n" + "=" * 70)
    print("V2 PAPER TRADING SIMULATOR")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print(f"Output file: {COMPARISON_LOG}")
    print(f"Interval: Check every 60 seconds")
    print("\nRun alongside: python main.py (V1)")
    print("=" * 70 + "\n")
    
    last_check = None
    
    while True:
        try:
            # Get latest OHLCV data
            df = get_ohlcv(SYMBOL)
            if df.empty:
                time.sleep(60)
                continue
            
            current_time = pd.to_datetime(df.index[-1]) if hasattr(df.index[-1], '__class__') else datetime.now()
            
            # Only process on new candle
            if last_check and current_time <= last_check:
                time.sleep(60)
                continue
            
            last_check = current_time
            
            # Read V1 last decision (from trading_log.csv)
            v1_decision = None
            if Path("trading_log.csv").exists():
                try:
                    v1_log = pd.read_csv("trading_log.csv").tail(1)
                    if len(v1_log) > 0:
                        v1_decision = {
                            "signal": "buy" if v1_log['side'].iloc[0] == "Buy" else "sell",
                            "qty": float(v1_log['qty'].iloc[0]),
                            "confidence": float(v1_log.get('confidence', 50).iloc[0]),
                            "take_profit": float(v1_log['take_profit'].iloc[0]),
                            "stop_loss": float(v1_log['stop_loss'].iloc[0]),
                        }
                except:
                    pass
            
            # Simulate V2 decision
            v2_decision = simulate_v2_decision(df)
            
            # Log comparison
            if v1_decision or v2_decision:
                comparison = log_comparison(v1_decision, v2_decision)
                
                if v1_decision and v2_decision:
                    agreement = "✅" if comparison['signal_agreement'] else "⚠️"
                    print(f"{agreement} [{current_time}] V1:{v1_decision['signal'].upper()} "
                          f"({v1_decision['confidence']:.0f}%) | "
                          f"V2:{v2_decision['signal'].upper()} "
                          f"({v2_decision['confidence']:.0f}%)")
                elif v2_decision and not v1_decision:
                    print(f"💡 [{current_time}] V2 found edge: {v2_decision['signal'].upper()} "
                          f"({v2_decision['confidence']:.0f}%) [V1 missed]")
                elif v1_decision and not v2_decision:
                    print(f"ⓘ  [{current_time}] V1 entered but V2 filtered (low confidence)")
            
            time.sleep(60)  # Check every minute
            
        except KeyboardInterrupt:
            print_comparison_summary()
            print("\n📊 Comparison complete. Analyze v1_vs_v2_comparison.csv")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()
