# ml/merge_datasets.py
import os
import sys
import pandas as pd
from pathlib import Path

# Configurações de caminhos
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.config import FEATURES

LOG_FILE = os.path.join(ROOT, "trading_log.csv")
BACKTEST_FILE = os.path.join(ROOT, "ml", "dataset.csv")

def process_production_logs():
    if not os.path.exists(LOG_FILE):
        print(f"⚠ Arquivo de produção não encontrado: {LOG_FILE}")
        return pd.DataFrame()
        
    df_log = pd.read_csv(LOG_FILE, on_bad_lines='skip')
    
    if df_log.empty:
        return pd.DataFrame()

    print(f"📊 Lendo {len(df_log)} trades do ambiente de produção...")
    
    # 1. Criação do 'label' (Gabarito de acertos/erros)
    # Garante que o pnl seja número e cria a label de lucro (1) ou prejuízo (0)
    df_log['pnl'] = pd.to_numeric(df_log['pnl'], errors='coerce')
    df_log = df_log.dropna(subset=['pnl']) 
    df_log['label'] = (df_log['pnl'] > 0).astype(int)
    
    # 2. Tradução de variáveis (Feature Engineering)
    risk_map = {"baixo": 0, "medio": 1, "alto": 2, "desconhecido": 1}
    df_log['risk_level_encoded'] = df_log['risk_level'].map(risk_map).fillna(1)
    
    # Na produção, o preço de entrada (entry_price) equivale ao 'close' daquele momento
    if 'entry_price' in df_log.columns:
        df_log['close'] = df_log['entry_price']
        
    # 3. Alinhamento de Colunas com o modelo ML
    cols_to_keep = FEATURES + ['label']
    for col in cols_to_keep:
        if col not in df_log.columns:
            # Preenche colunas que não existiam na produção com 0 para não quebrar o algoritmo
            df_log[col] = 0.0 
            
    return df_log[cols_to_keep].copy()

def merge():
    df_prod = process_production_logs()
    
    if not os.path.exists(BACKTEST_FILE):
        print(f"❌ Arquivo de backtest não encontrado: {BACKTEST_FILE}")
        return
        
    df_backtest = pd.read_csv(BACKTEST_FILE)
    print(f"📈 Lendo {len(df_backtest)} trades gerados pelo backtest...")
    
    if not df_prod.empty:
        df_merged = pd.concat([df_backtest, df_prod], ignore_index=True)
        print(f"✅ Fusão concluída com sucesso! Dataset unificado: {len(df_merged)} trades.")
    else:
        print("⚠ Nenhum trade de produção processado. O dataset continua apenas com o backtest.")
        df_merged = df_backtest
        
    # Segurança: Faz backup do dataset atual antes de sobrescrever
    backup_path = os.path.join(ROOT, "ml", "dataset_backtest_bkp.csv")
    if not os.path.exists(backup_path):
        os.rename(BACKTEST_FILE, backup_path)
        print(f"💾 Backup de segurança salvo em: {backup_path}")
        
    # Salva o arquivo final no formato que o train_model.py espera ler
    df_merged.to_csv(BACKTEST_FILE, index=False)
    print(f"🚀 Base de dados pronta para treinamento injetada em: {BACKTEST_FILE}")

if __name__ == "__main__":
    merge()