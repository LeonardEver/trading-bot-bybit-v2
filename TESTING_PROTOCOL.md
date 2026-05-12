# TESTING METHODOLOGY: Validate Improvements Before Deployment
## Date: April 19, 2026 | Validation Protocol

---

## OVERVIEW
This document outlines how to **safely test** each corrected function against the current bot to prove edge improvement without risking real capital.

---

## PHASE 1: PAPER TRADING (1-2 weeks)

### SetUp: Run V1 (Current) and V2 (Corrected) in Parallel

```bash
# Create two separate processes
Terminal 1: python main.py                   # Current version (LIVE on Bybit testnet)
Terminal 2: python main_v2_corrected.py      # Corrected version (Paper trading sim)
```

### Metrics to Track in Parallel

| Metric | V1 Current | V2 Corrected | Target | Pass/Fail |
|--------|-----------|-------------|--------|-----------|
| Win Rate | ? | ? | >45% | ✓/✗ |
| Profit Factor | ? | ? | >1.5 | ✓/✗ |
| Sharpe Ratio | ? | ? | >0.5 | ✓/✗ |
| Drawdown Max | ? | ? | <-$40 | ✓/✗ |
| Avg Win:Loss | ? | ? | >1:5 | ✓/✗ |

### Data Collection

```python
# Create unified logging for comparison
def log_comparison_metrics(trade_v1, trade_v2):
    """
    Compare both versions' decision on same candle
    """
    comparison = {
        "timestamp": datetime.now(),
        "candle_close": trade_v1["entry_price"],
        
        # V1 Metrics
        "v1_signal": trade_v1.get("signal"),
        "v1_qty": trade_v1.get("qty"),
        "v1_confidence": trade_v1.get("confidence"),
        "v1_pnl": trade_v1.get("pnl"),
        
        # V2 Metrics
        "v2_signal": trade_v2.get("signal"),
        "v2_qty": trade_v2.get("qty"),
        "v2_confidence": trade_v2.get("confidence"),
        "v2_pnl": trade_v2.get("pnl"),
        
        # Deltas
        "signal_agreement": 1 if trade_v1.get("signal") == trade_v2.get("signal") else 0,
        "pnl_delta": trade_v2.get("pnl", 0) - trade_v1.get("pnl", 0),
    }
    
    df_comparison = pd.DataFrame([comparison])
    df_comparison.to_csv("v1_vs_v2_comparison.csv", mode='a', index=False)
    
    return comparison
```

### Weekly Report Template

```markdown
## WEEK 1 Report (April 21-27, 2026)

### V1 (Current Bot)
- Trades executed: 45
- Win rate: 24% (11/45)
- Total PnL: -$12.50
- Sharpe: -0.35

### V2 (Corrected Bot)
- Trades executed: 38  # Fewer due to stricter filters
- Win rate: 52% (20/38)
- Total PnL: +$18.75
- Sharpe: +1.12

### Key Observations
- V2 enters fewer trades but with higher quality
- V2 exits earlier (TP/SL adjusted)
- V2 avoids revenge trades (cooldown logic)

### Next Week Actions
- [ ] Tune threshold if too conservative
- [ ] Validate across ETHUSDT
```

---

## PHASE 2: Walk-Forward Validation (Historical Data)

### Backtest V1 vs V2 on Historical Data

```python
# backtest_validator.py
import pandas as pd
from datetime import datetime, timedelta

def backtest_versions(start_date='2026-02-01', end_date='2026-03-31', walk_forward_window=14):
    """
    Split historical data and validate both versions
    """
    
    # Load historical candles
    df_historical = load_historical_data("BTCUSDT", start_date, end_date)
    
    all_results = []
    
    # Walk-forward: Train on 60 days, test on next 14 days
    test_start = start_date + timedelta(days=60)
    
    while test_start < end_date:
        test_end = min(test_start + timedelta(days=walk_forward_window), end_date)
        train_end = test_start - timedelta(days=1)
        train_start = train_end - timedelta(days=60)
        
        print(f"\n=== TESTING: {test_start.date()} to {test_end.date()} ===")
        
        # Get test window data
        df_test = df_historical[(df_historical.index >= test_start) & 
                                (df_historical.index <= test_end)]
        
        # Simulate V1 on test window
        results_v1 = simulate_bot_v1(df_test)
        stats_v1 = calculate_statistics(results_v1)
        
        # Simulate V2 on test window
        results_v2 = simulate_bot_v2_corrected(df_test)
        stats_v2 = calculate_statistics(results_v2)
        
        # Compare
        print(f"V1 → Win Rate: {stats_v1['win_rate']:.1%} | Sharpe: {stats_v1['sharpe']:.2f} | PnL: ${stats_v1['total_pnl']:.2f}")
        print(f"V2 → Win Rate: {stats_v2['win_rate']:.1%} | Sharpe: {stats_v2['sharpe']:.2f} | PnL: ${stats_v2['total_pnl']:.2f}")
        
        all_results.append({
            "period": f"{test_start.date()}-{test_end.date()}",
            "v1_wr": stats_v1['win_rate'],
            "v2_wr": stats_v2['win_rate'],
            "v1_sharpe": stats_v1['sharpe'],
            "v2_sharpe": stats_v2['sharpe'],
            "v1_pnl": stats_v1['total_pnl'],
            "v2_pnl": stats_v2['total_pnl'],
            "improvement_wr_pct": (stats_v2['win_rate'] - stats_v1['win_rate']) * 100,
            "improvement_pnl": stats_v2['total_pnl'] - stats_v1['total_pnl'],
        })
        
        test_start += timedelta(days=walk_forward_window)
    
    # Summary
    df_results = pd.DataFrame(all_results)
    
    print("\n=== WALK-FORWARD SUMMARY ===")
    print(df_results.to_string(index=False))
    
    print(f"\nAverage Win Rate Improvement: {df_results['improvement_wr_pct'].mean():.1f}pp")
    print(f"Average PnL Improvement: ${df_results['improvement_pnl'].mean():.2f}")
    print(f"Win Rate Improvement Consistency: {(df_results['improvement_wr_pct'] > 0).sum() / len(df_results) * 100:.0f}% of periods positive")
    
    # Only approve if:
    approval_criteria = {
        "avg_wr_improvement > 10pp": df_results['improvement_wr_pct'].mean() > 10,
        "wr_positive_in > 70% of periods": (df_results['improvement_wr_pct'] > 0).sum() / len(df_results) > 0.70,
        "sharpe_improvement > 0.5": (df_results['v2_sharpe'] - df_results['v1_sharpe']).mean() > 0.5,
        "no_sharpe_degradation": (df_results['v2_sharpe'] < df_results['v1_sharpe']).sum() == 0,
    }
    
    print("\n=== APPROVAL CRITERIA ===")
    for criterion, result in approval_criteria.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | {criterion}")
    
    if all(approval_criteria.values()):
        print("\n🚀 V2 APPROVED FOR DEPLOYMENT")
        return True
    else:
        print("\n⚠️ V2 REQUIRES ADJUSTMENTS")
        return False
```

### Run Walk-Forward Validation

```bash
python backtest_validator.py

# Expected output:
# === TESTING: 2026-03-01 to 2026-03-14 ===
# V1 → Win Rate: 28.6% | Sharpe: -0.42 | PnL: -$8.50
# V2 → Win Rate: 52.4% | Sharpe: +1.15 | PnL: +$22.75
#
# === WALK-FORWARD SUMMARY ===
# Average Win Rate Improvement: +18.5pp
# Average PnL Improvement: +$18.33
# Win Rate Improvement Consistency: 100% of periods positive
#
# === APPROVAL CRITERIA ===
# ✅ PASS | avg_wr_improvement > 10pp
# ✅ PASS | wr_positive_in > 70% of periods
# ✅ PASS | sharpe_improvement > 0.5
# ✅ PASS | no_sharpe_degradation
#
# 🚀 V2 APPROVED FOR DEPLOYMENT
```

---

## PHASE 3: Robustness Testing

### Test 1: Regime Change Performance
```python
def test_regime_switch():
    """
    Verify V2 adapts to different market conditions
    """
    
    # Trending market (consecutive higher highs/lows)
    df_trending = load_historical_data_filtered("2026-03-11", "2026-03-14")  # Strong uptrend
    results_trending = simulate_bot_v2_corrected(df_trending)
    
    # Ranging market
    df_ranging = load_historical_data_filtered("2026-03-17", "2026-03-20")   # Consolidation
    results_ranging = simulate_bot_v2_corrected(df_ranging)
    
    # High volatility
    df_highvol = load_historical_data_filtered("2026-03-18", "2026-03-19")   # Flash crash
    results_highvol = simulate_bot_v2_corrected(df_highvol)
    
    print(f"Trending market WR: {calculate_statistics(results_trending)['win_rate']:.1%}")
    print(f"Ranging market WR: {calculate_statistics(results_ranging)['win_rate']:.1%}")
    print(f"High vol market WR: {calculate_statistics(results_highvol)['win_rate']:.1%}")
    
    # All should be > 40%
```

### Test 2: Edge Cases
```python
def test_edge_cases():
    """
    Handle unusual market conditions without errors
    """
    
    test_cases = [
        ("Gap down open", "<EURUSD missed data>"),
        ("Circuit breaker halt", "<Market closed>"),
        ("Extreme volatility spike", "<ATR = +300%>"),
        ("Low liquidity", "<Spread = 5 points>"),
    ]
    
    for scenario, market_condition in test_cases:
        try:
            result = simulate_bot_v2_corrected(market_condition)
            print(f"✅ {scenario}: Handled gracefully")
        except Exception as e:
            print(f"❌ {scenario}: {e}")
```

### Test 3: Multi-Symbol Validation
```python
def test_multi_symbol():
    """
    Verify corrected bot works across BTC and ETH
    """
    
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        df = load_historical_data(symbol, "2026-03-01", "2026-03-31")
        results = simulate_bot_v2_corrected(df)
        stats = calculate_statistics(results)
        
        print(f"{symbol}: WR={stats['win_rate']:.1%}, Sharpe={stats['sharpe']:.2f}")
        
        # Both should have positive edge
        if stats['sharpe'] > 0.3:
            print(f"  ✅ PASS: Edge detected")
        else:
            print(f"  ❌ FAIL: No edge")
```

---

## PHASE 4: Production Deployment Checklist

Before going LIVE on real trading account, verify:

- [ ] **Walk-forward validation passed** (all approval criteria met)
- [ ] **Paper trading results positive** for 7+ consecutive days
- [ ] **No errors in edge case testing**
- [ ] **Multi-symbol tested** (if deploying on BTC + ETH)
- [ ] **Logging verified** (all trades recorded with correct entry/exit)
- [ ] **Risk limits active** (daily loss limit, max drawdown, etc.)
- [ ] **Alert system functional** (webhook/email on errors)
- [ ] **Backup/rollback plan** (can revert to V1 in 5 min)

### Deployment Steps

```bash
# Step 1: Backup current
cp main.py main_v1_backup.py

# Step 2: Deploy V2
cp main_v2_corrected.py main.py

# Step 3: Start with REDUCED capital
# Edit config.py: RISK_PER_TRADE = 0.005  (0.5% instead of 1%)

# Step 4: Monitor first 24h
python main.py &
# Watch logs in real-time
tail -f trading_log.csv
# Alert on first loss > 3%

# Step 5: Scale up capital if positive
# After 1 week of positive results, increase to RISK_PER_TRADE = 0.01

# Step 6: Full deployment
# After 2 weeks of total PnL > $500, deploy on full capital
```

---

## EXPECTED OUTCOMES & VALIDATION THRESHOLDS

### Minimum Acceptance Criteria

| Metric | Current (V1) | Target (V2) | Pass Threshold |
|--------|-------------|-----------|-----------------|
| **Win Rate** | 26% | 50% | >45% |
| **Profit Factor** | 0.44 | 1.8 | >1.5 |
| **Sharpe Ratio** | -0.30 | +0.80 | >+0.3 |
| **Max Drawdown** | -$0.16 | -$0.05 | <-$0.08 |
| **Avg Win/Loss** | 1:95 | 1:10 | <1:8 |
| **Trades/Week** | 45 | 35 | 30-40 |

### Success Definition
✅ **V2 is approved for live deployment** if **ALL** of the following are true:
1. Win rate improvement: **+15pp minimum** (26% → 41%+)
2. Sharpe ratio: **Positive** (currently: -0.30)
3. Walk-forward validation: **Passes in 80%+ of test periods**
4. No black swan losses: **Single trade loss < 2% of capital**
5. Consistency: **3 consecutive weeks of positive monthly PnL**

---

## ROLLBACK PROCEDURE

If V2 shows signs of failure:

```bash
# Immediate action (within 5 min)
kill %1                           # Stop V2
cp main_v1_backup.py main.py      # Restore V1
python main.py &                  # Restart V1

# Analysis
# 1. Identify which component failed
# 2. Revert that component, keep others
# 3. Re-test in paper trading

# Example: If TP/SL regime detection bad
# → Revert to original multipliers but keep Kelly sizing
# → Keep better ML filter but revert sentiment cache
```

---

## MONITORING DASHBOARD (Live)

Display in real-time:

```
╔════════════════════════════════════════════════╗
║          BOT V2 PERFORMANCE MONITOR            ║
╠════════════════════════════════════════════════╣
║ Uptime: 14d 7h 22m                             ║
║ Trades Today: 3 | This Week: 24                ║
║ Win Rate: 52.4% (vs 27% baseline)  ████████░░ ║
║ Sharpe Ratio: +1.12 (vs -0.30)                ║
║ Total PnL: +$87.50 | Daily: +$6.25            ║
║ Drawdown: -$22.50 (Peak: -$28) [SAFE]         ║
║ Position: 0 (awaiting next signal)            ║
║ ML Model: Updated 2h ago | Acc: 0.58          ║
║ Last Trade: 2026-04-19 14:52:15 | +0.75%     ║
╚════════════════════════════════════════════════╝
```

---

## SIGN-OFF

**Ready to deploy V2 to production?** → Complete all phases and get approval from:
- [ ] Quant Researcher (me)
- [ ] Risk Officer (2% max daily loss confirmed)
- [ ] Operations (monitoring/alerting active)
- [ ] Finance (capital allocation approved)
