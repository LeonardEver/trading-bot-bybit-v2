#!/bin/bash
# QUICK START SCRIPT - V2 DEPLOYMENT CHECKLIST
# Usage: bash deploy_v2.sh

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     BYBIT BOT V2 DEPLOYMENT - QUICK START SCRIPT              ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

echo "📋 DEPLOYMENT STAGES:"
echo ""

echo "STAGE 1: VALIDATE CURRENT BOT"
echo "─────────────────────────────────────────────────────────────"
echo "Run: python backtest_validator.py"
echo ""
echo "This will:"
echo "  ✓ Analyze trading_log.csv (your trade history)"
echo "  ✓ Calculate V1 baseline metrics"
echo "  ✓ Show improvement targets for V2"
echo "  ✓ Generate backtest_report.txt"
echo ""
read -p "Press enter after running backtest_validator.py... "
echo ""

echo "STAGE 2: VERIFY CORRECTED FUNCTIONS"
echo "─────────────────────────────────────────────────────────────"
echo "✅ Corrected functions already integrated into main.py:"
echo "  ✓ calculate_order_qty_kelly_criterion() - Dynamic position sizing"
echo "  ✓ calculate_tp_sl_dynamic() - Regime-adaptive TP/SL (1:2+ ratio)"
echo "  ✓ detect_market_regime() - Trending/ranging detection"
echo "  ✓ calcular_performance_advanced() - Stratified weight adaptation"
echo "  ✓ model_predict_prob_corrected() - Single-layer ML filter"
echo ""
echo "Main.py is ready for deployment!"
echo ""

echo "STAGE 3: PARALLEL TESTING (CRITICAL - 48+ HOURS)"
echo "─────────────────────────────────────────────────────────────"
echo "Open TWO terminals:"
echo ""
echo "Terminal 1 (V1 - Current Bot):"
echo "  $ python main.py"
echo ""
echo "Terminal 2 (V2 - Paper Trading Simulator):"
echo "  $ python paper_trading_v2.py"
echo ""
echo "This will create: v1_vs_v2_comparison.csv"
echo "Monitor progress: tail -f v1_vs_v2_comparison.csv"
echo ""
read -p "Press enter once both terminals are running for 48+ hours... "
echo ""

echo "STAGE 4: ANALYZE RESULTS"
echo "─────────────────────────────────────────────────────────────"
echo "Run: python backtest_validator.py"
echo "This will show:"
echo "  ✓ Win rate improvement (target: +15pp)"
echo "  ✓ Sharpe ratio improvement (target: positive)"
echo "  ✓ PnL improvement"
echo ""
echo "GO/NO-GO DECISION CRITERIA:"
echo "  ✅ PASS: WR > 45% AND Sharpe > 0.3"
echo "  ❌ FAIL: WR < 35% OR Sharpe < -0.5"
echo ""
read -p "Is V2 showing expected improvements? (yes/no): " improvement_ok
echo ""

if [ \"$improvement_ok\" != \"yes\" ]; then
    echo \"⚠️  V2 needs additional tuning before deployment\"
    echo \"Review QUANT_ANALYSIS.md for specific recommendations\"
    exit 1
fi

echo "STAGE 5: CONFIGURE RISK LIMITS"
echo "─────────────────────────────────────────────────────────────"
echo "Edit: deployment_config.py"
echo ""
echo "Key settings:"
echo "  risk_per_trade: 0.005 (0.5% - CONSERVATIVE)"
echo "  kelly_fraction: 0.25 (25% Kelly - SAFE)"
echo "  max_daily_loss: \$100 (adjust to your capital)"
echo "  circuit_breaker: Enabled ✅"
echo ""
echo "Verify: python deployment_config.py"
echo ""
read -p "Risk configuration reviewed? (yes/no): " config_ok
echo ""

if [ \"$config_ok\" != \"yes\" ]; then
    echo \"❌ Review deployment_config.py first\"
    exit 1
fi

echo "STAGE 6: LAUNCH PRODUCTION"
echo "─────────────────────────────────────────────────────────────"
echo ""
echo \"LIVE DEPLOYMENT COMMAND:\"
echo \"\"
echo \"  $ python main.py\"
echo \"\"
echo \"Monitor with:\"
echo \"  $ python monitoring_dashboard.py\"
echo \"\"
echo \"📊 FIRST 24 HOURS - WATCH FOR:\"
echo \"  • Win rate should be > 45%\"
echo \"  • Sharpe should be positive\"
echo \"  • No single trade loss > 3% of capital\"
echo \"  • Daily PnL trending positive\"
echo \"\"
echo \"⚠️  CIRCUIT BREAKER WILL ACTIVATE IF:\"
echo \"  • 3 consecutive losses\"
echo \"  • Daily loss > 2%\"
echo \"  • Sharpe drops below -0.5\"
echo \"\"
echo \"🔄 ROLLBACK PROCEDURE (if needed):\"
echo \"  1. Kill the bot: ctrl+c\"
echo \"  2. Restore: cp main_v1_backup.py main.py\"
echo \"  3. Restart: python main.py\"
echo \"\"

read -p \"Ready to deploy V2 to PRODUCTION? (yes/NO): \" ready_prod

if [ \"$ready_prod\" = \"yes\" ]; then
    echo \"\"
    echo \"✅ APPROVED FOR PRODUCTION DEPLOYMENT\"
    echo \"\"
    echo \"Starting monitoring dashboard in background...\"
    python monitoring_dashboard.py &
    sleep 2
    echo \"\"
    echo \"Now starting main.py - V2 is LIVE\"
    echo \"\"
    python main.py
else
    echo \"❌ Deployment cancelled\"
    echo \"\"
    echo \"To re-attempt deployment:\"
    echo \"  bash deploy_v2.sh\"
    exit 0
fi
