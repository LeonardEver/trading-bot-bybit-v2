import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


DATA_CSV = ROOT / "ml" / "dataset.csv"
OUT_DIR = ROOT / "ml" / "alpha"


def summarize_forward_return(df, signal_mask, horizon=3):
    close = pd.to_numeric(df["close"], errors="coerce")
    forward_return = close.shift(-horizon) / close - 1
    selected = forward_return[signal_mask].dropna()
    if selected.empty:
        return {"count": 0, "mean_forward_return": 0.0, "hit_rate": 0.0}
    return {
        "count": int(len(selected)),
        "mean_forward_return": float(selected.mean()),
        "hit_rate": float((selected > 0).mean()),
    }


def discover_vwap_reversion(df):
    z = pd.to_numeric(df.get("vwap_deviation_zscore", 0.0), errors="coerce").fillna(0.0)
    high_extreme = z > 2.0
    low_extreme = z < -2.0
    return {
        "short_reversion_from_high_vwap_z": summarize_forward_return(df, high_extreme),
        "long_reversion_from_low_vwap_z": summarize_forward_return(df, low_extreme),
    }


def discover_cvd_divergence(df):
    divergence = pd.to_numeric(df.get("spot_perp_cvd_divergence", 0.0), errors="coerce").fillna(0.0)
    positive = divergence > divergence.rolling(200).quantile(0.95).fillna(np.inf)
    negative = divergence < divergence.rolling(200).quantile(0.05).fillna(-np.inf)
    return {
        "spot_stronger_than_perp": summarize_forward_return(df, positive),
        "perp_stronger_than_spot": summarize_forward_return(df, negative),
    }


def run_alpha_discovery(path: Path = DATA_CSV):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    report = {
        "vwap_reversion": discover_vwap_reversion(df),
        "cvd_spot_perp_divergence": discover_cvd_divergence(df),
    }
    (OUT_DIR / "alpha_discovery_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_alpha_discovery(), indent=2))
