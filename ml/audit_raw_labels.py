"""Quick check: raw triple barrier label distribution BEFORE NaN filtering."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from ml.ml_data_pipeline_v2 import triple_barrier_labels

df = pd.read_csv(ROOT / "dataset.csv")
print(f"Total rows: {len(df):,}")

labels = triple_barrier_labels(df)
tp = (labels == 1).sum()
sl = (labels == 0).sum()
neutral = labels.isna().sum()
total = len(labels)

print(f"TP primeiro:     {tp:,} ({tp/total*100:.1f}%)")
print(f"SL primeiro:     {sl:,} ({sl/total*100:.1f}%)")
print(f"Nenhuma (NaN):   {neutral:,} ({neutral/total*100:.1f}%)")
print(f"Total:           {total:,}")
print(f"Ratio TP/SL:     {tp/sl:.3f}")
print(f"Label rate:      {(tp+sl)/total*100:.1f}%")
print(f"Discard rate:    {neutral/total*100:.1f}%")
