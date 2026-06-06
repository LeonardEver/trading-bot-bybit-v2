# ml/ml_data_pipeline_v2.py
"""
Triple Barrier Method — pipeline de dados v2.

Diferença vs. v1 (ml_data_pipeline.py):
  - Label: Triple Barrier Method (TP +0.40% / SL -0.20% / Time 12 candles)
    em vez de direção simples do candle.
  - Features, lag estrito, e todo o resto são IDÊNTICOS ao v1.

Regras do plano TRIPLE_BARRIER.md:
  - NÃO adicionar novas features
  - NÃO alterar hiperparâmetros
  - Preservar pipeline atual para comparação A/B
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from ml.config import FEATURES
from ml.features import apply_strict_feature_lag, prepare_features
from ml.ml_data_pipeline import (
    fetch_ohlcv,
    attach_sentiment,
    assign_risk_level,
    to_timestamp_ms,
)

# ---------------------------------------------------------------------------
# Parâmetros do Triple Barrier (conforme TRIPLE_BARRIER.md)
# ---------------------------------------------------------------------------
UPPER_BARRIER = +0.0040   # +0.40% Take Profit
LOWER_BARRIER = -0.0020   # -0.20% Stop Loss
TIME_BARRIER  = 12         # 12 candles (60 minutos em TF=5)

SYMBOL = "BTCUSDT"
TF = "5"
LOOKBACK = 100000

OUT_CSVS = [
    ROOT / "dataset_triple_barrier.csv",
    ROOT / "ml" / "dataset_triple_barrier.csv",
]


# ---------------------------------------------------------------------------
# Triple Barrier Labeling
# ---------------------------------------------------------------------------
def triple_barrier_labels(
    df: pd.DataFrame,
    upper: float = UPPER_BARRIER,
    lower: float = LOWER_BARRIER,
    time_barrier: int = TIME_BARRIER,
) -> pd.Series:
    """
    Para cada candle t (linha), determina qual barreira é atingida primeiro
    nas velas futuras t+1 ... t+time_barrier.

    Retorna pd.Series alinhada ao índice de df:
        1  = Take Profit (upper) atingido antes do Stop Loss
        0  = Stop Loss (lower) atingido antes do Take Profit
        NaN = barreira temporal expirou sem atingir nenhuma barreira
    """
    n = len(df)
    labels = np.full(n, np.nan, dtype=float)

    close_arr = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    high_arr  = pd.to_numeric(df["high"],  errors="coerce").to_numpy(dtype=float)
    low_arr   = pd.to_numeric(df["low"],   errors="coerce").to_numpy(dtype=float)
    open_arr  = pd.to_numeric(df["open"],  errors="coerce").to_numpy(dtype=float)

    for t in range(n):
        entry_price = close_arr[t]
        if np.isnan(entry_price) or entry_price <= 0:
            continue

        tp_price = entry_price * (1.0 + upper)
        sl_price = entry_price * (1.0 + lower)

        hit = False
        for k in range(t + 1, min(t + 1 + time_barrier, n)):
            h = high_arr[k]
            l = low_arr[k]
            o = open_arr[k]
            c = close_arr[k]

            if np.isnan(h) or np.isnan(l):
                continue

            upper_hit = (h >= tp_price)
            lower_hit = (l <= sl_price)

            if upper_hit and not lower_hit:
                labels[t] = 1.0
                hit = True
                break
            elif lower_hit and not upper_hit:
                labels[t] = 0.0
                hit = True
                break
            elif upper_hit and lower_hit:
                # Ambos atingidos na mesma vela — usa direção intrabar
                # como proxy de qual foi atingido primeiro.
                if c >= o:
                    labels[t] = 1.0
                else:
                    labels[t] = 0.0
                hit = True
                break

        # Se não atingiu nenhuma barreira, labels[t] permanece NaN

    return pd.Series(labels, index=df.index, dtype=float)


# ---------------------------------------------------------------------------
# Construção do dataset causal com Triple Barrier
# ---------------------------------------------------------------------------
def build_causal_dataset_triple_barrier(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline v2 completo:
      1. Prepara features (idêntico ao v1)
      2. Cria labels via Triple Barrier Method
      3. Aplica lag estrito nas features (idêntico ao v1)
      4. Remove linhas com label NaN ou features ausentes
    """
    df = raw_df.copy()
    df["timestamp"] = to_timestamp_ms(df["timestamp"])

    # Features — idêntico ao v1
    df = prepare_features(df)

    # Label Triple Barrier
    df["label"] = triple_barrier_labels(df)

    # Lag estrito — idêntico ao v1
    df = apply_strict_feature_lag(df, FEATURES, periods=1)

    # Remove linhas onde label é NaN ou features ausentes
    df = df.dropna(subset=FEATURES + ["label"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Regeneração a partir de CSV existente
# ---------------------------------------------------------------------------
def regenerate_from_csv(source_csv: Path | str) -> pd.DataFrame:
    """Re-processa um CSV de OHLCV existente com Triple Barrier labels."""
    source_csv = Path(source_csv)
    if not source_csv.exists():
        raise FileNotFoundError(f"Dataset fonte não encontrado: {source_csv}")

    raw_df = pd.read_csv(source_csv)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(raw_df.columns)
    if missing:
        raise ValueError(
            f"Dataset fonte sem colunas obrigatórias: {sorted(missing)}"
        )

    if "sentiment_score" not in raw_df.columns:
        raw_df["sentiment_score"] = 0.0
    if "risk_level_encoded" not in raw_df.columns:
        raw_df["risk_level_encoded"] = 1

    cleaned = build_causal_dataset_triple_barrier(raw_df)
    for out_csv in OUT_CSVS:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(out_csv, index=False)
    return cleaned


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Triple Barrier Pipeline v2")
    print(f"  Upper : {UPPER_BARRIER:+.4f} ({UPPER_BARRIER*100:+.2f}%)")
    print(f"  Lower : {LOWER_BARRIER:+.4f} ({LOWER_BARRIER*100:+.2f}%)")
    print(f"  Time  : {TIME_BARRIER} candles")
    print("=" * 60)

    # Tenta usar dataset OHLCV existente primeiro (mais rápido)
    source = ROOT / "dataset.csv"
    if source.exists():
        print(f"\n[INFO] Dataset fonte encontrado: {source}")
        print("Regenerando com Triple Barrier labels...")
        try:
            df = regenerate_from_csv(source)
            print(f"[OK] Dataset Triple Barrier gerado: {len(df)} linhas")
            label_dist = df["label"].value_counts().to_dict()
            print(f"      Distribuição labels: {label_dist}")
            return
        except Exception as e:
            print(f"[WARN] Falha ao regenerar do CSV: {e}")
            print("       Tentando baixar OHLCV fresco...")

    # Fallback: baixa OHLCV fresco
    print("\nBaixando OHLCV...")
    df = fetch_ohlcv(SYMBOL, interval=TF, limit=LOOKBACK)
    if df.empty:
        print("Erro: nenhum dado OHLCV retornado.")
        return

    print("Anexando sentimento (pode demorar)...")
    try:
        df = attach_sentiment(df)
    except Exception as e:
        print("Erro no sentiment attach:", e)
        df["sentiment_score"] = 0.0

    print("Atribuindo nível de risco...")
    df = assign_risk_level(df)

    print("Calculando features e labels Triple Barrier...")
    try:
        df = build_causal_dataset_triple_barrier(df)
    except Exception as e:
        print("Erro ao construir dataset Triple Barrier:", e)
        import traceback
        traceback.print_exc()
        return

    for out_csv in OUT_CSVS:
        print(f"Salvando dataset em {out_csv}")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

    label_dist = df["label"].value_counts().to_dict()
    print(f"\nConcluído. {len(df)} linhas | Labels: {label_dist}")


if __name__ == "__main__":
    main()
