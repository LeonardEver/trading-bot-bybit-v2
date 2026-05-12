"""
MONITORING DASHBOARD - Live Performance Tracker
Displays real-time metrics for V2 deployment
Run in separate terminal: python monitoring_dashboard.py
"""

import pandas as pd
import os
from pathlib import Path
from datetime import datetime, timedelta
import time

LOG_FILE = "trading_log.csv"


def get_daily_stats(df):
    """Calculate stats for current day"""
    today = df[pd.to_datetime(df['timestamp']).dt.date == datetime.today().date()]
    if len(today) == 0:
        return {
            "trades_today": 0,
            "pnl_today": 0.0,
            "wins_today": 0,
            "losses_today": 0,
            "win_rate_today": 0.0
        }
    
    pnls = today['pnl'].astype(float).values
    wins = len(pnls[pnls > 0])
    losses = len(pnls[pnls < 0])
    
    return {
        "trades_today": len(today),
        "pnl_today": pnls.sum(),
        "wins_today": wins,
        "losses_today": losses,
        "win_rate_today": wins / (wins + losses) if (wins + losses) > 0 else 0
    }


def get_weekly_stats(df):
    """Calculate stats for current week"""
    week_start = datetime.today() - timedelta(days=datetime.today().weekday())
    week_df = df[pd.to_datetime(df['timestamp']) >= week_start]
    
    if len(week_df) == 0:
        return {
            "trades_week": 0,
            "pnl_week": 0.0,
            "win_rate_week": 0.0
        }
    
    pnls = week_df['pnl'].astype(float).values
    wins = len(pnls[pnls > 0])
    losses = len(pnls[pnls < 0])
    
    return {
        "trades_week": len(week_df),
        "pnl_week": pnls.sum(),
        "win_rate_week": wins / (wins + losses) if (wins + losses) > 0 else 0
    }


def get_overall_stats(df):
    """Calculate cumulative stats"""
    if len(df) == 0:
        return {
            "total_trades": 0,
            "total_pnl": 0.0,
            "win_rate_all": 0.0,
            "sharpe_all": 0.0,
            "max_dd": 0.0
        }
    
    pnls = df['pnl'].astype(float).values
    wins = len(pnls[pnls > 0])
    losses = len(pnls[pnls < 0])
    
    # Sharpe
    returns = pnls / 100
    sharpe = (returns.mean() / returns.std()) * (252 ** 0.5) if returns.std() > 0 else 0
    
    # Max drawdown
    cumulative = pnls.cumsum()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_dd = drawdown.min() if len(drawdown) > 0 else 0
    
    return {
        "total_trades": len(df),
        "total_pnl": pnls.sum(),
        "win_rate_all": wins / (wins + losses) if (wins + losses) > 0 else 0,
        "sharpe_all": sharpe,
        "max_dd": max_dd
    }


def print_dashboard(df_stats, daily, weekly, overall):
    """Print formatted dashboard"""
    
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print("""
╔════════════════════════════════════════════════════════════════╗
║                  BYBIT BOT V2 MONITORING DASHBOARD             ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"📊 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # TODAY'S PERFORMANCE
    print("╔════ TODAY'S PERFORMANCE ═══════════════════════════╗")
    print(f"║ Trades:          {str(daily['trades_today']).rjust(5)}                      ║")
    print(f"║ Win Rate:        {(daily['win_rate_today']*100):5.1f}%  ({daily['wins_today']}/{daily['wins_today']+daily['losses_today']})              ║")
    print(f"║ PnL Today:       ${daily['pnl_today']:7.2f}  {'✅' if daily['pnl_today'] > 0 else '⚠️ '}                ║")
    print("╚═════════════════════════════════════════════════════╝")
    print()
    
    # WEEKLY PERFORMANCE
    print("╔════ WEEKLY PERFORMANCE ════════════════════════════╗")
    print(f"║ Trades:          {str(weekly['trades_week']).rjust(5)}                      ║")
    print(f"║ Win Rate:        {(weekly['win_rate_week']*100):5.1f}%                       ║")
    print(f"║ PnL Week:        ${weekly['pnl_week']:7.2f}  {'✅' if weekly['pnl_week'] > 0 else '⚠️ '}                ║")
    print("╚═════════════════════════════════════════════════════╝")
    print()
    
    # OVERALL METRICS
    print("╔════ OVERALL PERFORMANCE ════════════════════════════╗")
    print(f"║ Total Trades:    {str(overall['total_trades']).rjust(5)}                      ║")
    print(f"║ Win Rate:        {(overall['win_rate_all']*100):5.1f}%                       ║")
    print(f"║ Total PnL:       ${overall['total_pnl']:7.2f}  {'✅' if overall['total_pnl'] > 0 else '❌'}               ║")
    print(f"║ Sharpe Ratio:    {overall['sharpe_all']:5.2f}    {'✅' if overall['sharpe_all'] > 0.3 else '⚠️ '}               ║")
    print(f"║ Max Drawdown:    ${overall['max_dd']:7.2f}                  ║")
    print("╚═════════════════════════════════════════════════════╝")
    print()
    
    # ALERTS
    print("╔════ SYSTEM ALERTS ══════════════════════════════════╗")
    
    alerts = []
    
    if daily['pnl_today'] < -50:
        alerts.append(f"🚨 Large daily loss: ${daily['pnl_today']:.2f}")
    
    if daily['win_rate_today'] > 0 and daily['win_rate_today'] < 0.20:
        alerts.append(f"⚠️  Low daily win rate: {daily['win_rate_today']*100:.0f}%")
    
    if overall['sharpe_all'] < -0.5:
        alerts.append("⚠️  Sharpe ratio degrading")
    
    if len(alerts) == 0:
        alerts.append("✅ All systems operational")
    
    for alert in alerts:
        print(f"║ {alert:<50} ║")
    
    print("╚═════════════════════════════════════════════════════╝")
    print()
    
    # RECOMMENDATIONS
    print("╔════ TRENDING ═══════════════════════════════════════╗")
    if overall['total_trades'] < 20:
        print("║ Status: WARM UP - Insufficient data for evaluation   ║")
    elif overall['win_rate_all'] > 0.50 and overall['sharpe_all'] > 0.5:
        print("║ Status: ✅ STRONG - Bot performing as expected        ║")
    elif overall['win_rate_all'] > 0.45 and overall['sharpe_all'] > 0.0:
        print("║ Status: ✅ RUNNING - Bot performing adequately        ║")
    elif overall['win_rate_all'] < 0.40:
        print("║ Status: ⚠️  REVIEW - Consider rollback to V1           ║")
    else:
        print("║ Status: ⓘ  MONITOR - Continue observation tracking    ║")
    print("╚═════════════════════════════════════════════════════╝")
    print()
    
    print("💡 View live trades in: trading_log.csv")
    print("📈 Weekly reports generated: backtest_report.txt")
    print()


def main():
    """Main dashboard loop"""
    
    print("Starting monitoring dashboard...")
    
    while True:
        try:
            if not Path(LOG_FILE).exists():
                print(f"Waiting for {LOG_FILE} to be created...")
                time.sleep(10)
                continue
            
            # Read trading log
            df = pd.read_csv(LOG_FILE)
            
            if len(df) == 0:
                print("Waiting for first trade...")
                time.sleep(10)
                continue
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Calculate stats
            daily = get_daily_stats(df)
            weekly = get_weekly_stats(df)
            overall = get_overall_stats(df)
            
            # Print dashboard
            print_dashboard(df, daily, weekly, overall)
            
            # Refresh every 30 seconds
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n✅ Dashboard stopped")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    import numpy as np
    main()
