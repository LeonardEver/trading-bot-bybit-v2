# ml/backtest_data_generator.py
import os
import sys
import time
import pandas as pd
import requests
from pathlib import Path

# Garante que a raiz do projeto está no path para importar os seus módulos
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.technical_indicators import calculate_indicators
from strategies.strategy import generate_trade_signal
from ml.config import FEATURES

SYMBOL = "BTCUSDT"
INTERVAL = "15"
CANDLES_TO_FETCH = 200000  # 20.000 velas de 15m = aprox. 200 dias de histórico
DATASET_PATH = os.path.join(ROOT, "ml", "dataset.csv")

# Multiplicadores da sua estratégia (idênticos ao main.py)
SL_MULTIPLIER = 1.5
TP_MULTIPLIER = 2.0
TAXA_CORRETORA = 0.0010  # 0.10% total (0.05% entrada + 0.05% saída)

def fetch_historical_data(symbol, interval, total_candles):
    """Descarrega dados históricos massivos da Bybit usando paginação"""
    print(f"🔄 A descarregar {total_candles} velas históricas de {symbol} ({interval}m)...")
    url = "https://api.bybit.com/v5/market/kline"
    all_klines = []
    end_time = int(time.time() * 1000)

    while len(all_klines) < total_candles:
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": 1000,
            "end": end_time
        }
        
        try:
            resp = requests.get(url, params=params).json()
            if "result" not in resp or not resp["result"]["list"]:
                break
                
            chunk = resp["result"]["list"]
            all_klines.extend(chunk)
            
            # O último elemento é o mais antigo. Subtraímos 1ms para a próxima página
            end_time = int(chunk[-1][0]) - 1 
            
            print(f"   ↳ {len(all_klines)} velas descarregadas...")
            time.sleep(0.1) # Pausa para não bloquear a API
            
        except Exception as e:
            print(f"❌ Erro na API: {e}")
            break

    # Organiza do mais antigo para o mais recente
    df = pd.DataFrame(all_klines, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
        
    return df

def generate_backtest_dataset():
    # 1. Obter dados brutos
    df = fetch_historical_data(SYMBOL, INTERVAL, CANDLES_TO_FETCH)
    if df.empty:
        print("❌ Nenhum dado histórico recuperado.")
        return

    # 2. Calcular indicadores técnicos em todo o histórico de uma só vez (muito mais rápido)
    print("📈 A calcular indicadores técnicos...")
    df = calculate_indicators(df)
    
    # Adicionar variáveis de tempo
    dt = pd.to_datetime(df['timestamp'], unit='ms')
    df['hour'] = dt.dt.hour
    df['minute'] = dt.dt.minute
    
    # Mock do sentimento (0.0 = neutro) para o passado
    df['sentiment_score'] = 0.0 

    # 3. Máquina do Tempo: Simular operações
    print("🚀 A iniciar Máquina do Tempo (Simulação de Trades)...")
    dataset_rows = []
    
    # Começamos na linha 200 para ter histórico suficiente para as médias móveis (EMA200)
    for i in range(200, len(df) - 50):
        if i % 1000 == 0:
            print(f"   ↳ A processar vela {i} de {len(df)}...")

        # Passamos a "fotografia" do mercado até este exato momento
        window = df.iloc[i-100:i+1].copy() 
        
        # O seu ficheiro strategy.py gera o sinal
        trade_decision = generate_trade_signal(window)
        sinal = trade_decision.get("signal")
        
        if sinal in ["buy", "sell"]:
            entry_price = window.iloc[-1]['close']
            atr_atual = window.iloc[-1]['atr']
            
            if pd.isna(atr_atual) or atr_atual == 0:
                continue

            # Calcula Alvos Dinâmicos
            if sinal == "buy":
                tp = entry_price + (atr_atual * TP_MULTIPLIER)
                sl = entry_price - (atr_atual * SL_MULTIPLIER)
            else:
                tp = entry_price - (atr_atual * TP_MULTIPLIER)
                sl = entry_price + (atr_atual * SL_MULTIPLIER)

            # --- O GABARITO (Espreitar para o futuro) ---
            # Verificamos as próximas 50 velas para ver o que bate primeiro
            label = 0
            pnl_bruto_pct = 0.0
            
            for j in range(i+1, min(i+50, len(df))):
                fut_high = df.iloc[j]['high']
                fut_low = df.iloc[j]['low']
                
                if sinal == "buy":
                    if fut_low <= sl: # Bateu no Stop Loss
                        pnl_bruto_pct = (sl - entry_price) / entry_price
                        break
                    if fut_high >= tp: # Bateu no Take Profit
                        pnl_bruto_pct = (tp - entry_price) / entry_price
                        break
                elif sinal == "sell":
                    if fut_high >= sl: # Bateu no Stop Loss
                        pnl_bruto_pct = (entry_price - sl) / entry_price
                        break
                    if fut_low <= tp: # Bateu no Take Profit
                        pnl_bruto_pct = (entry_price - tp) / entry_price
                        break

            # Debita as taxas da Bybit
            pnl_liquido_pct = pnl_bruto_pct - TAXA_CORRETORA
            
            # Se deu lucro líquido, é um caso de sucesso (1). Se deu prejuízo, fracasso (0)
            label = 1 if pnl_liquido_pct > 0 else 0

            # --- Recolher Features para o Machine Learning ---
            row_data = {}
            for f in FEATURES:
                row_data[f] = df.iloc[i].get(f, 0.0)
                
            row_data["label"] = label
            row_data["pnl"] = round(pnl_liquido_pct * 100, 3) # Guardar percentagem para curiosidade
            row_data["side"] = sinal
            
            dataset_rows.append(row_data)

    # 4. Exportar o Dataset Massivo
    if dataset_rows:
        df_dataset = pd.DataFrame(dataset_rows)
        # Manter apenas as colunas que o ML precisa + a label
        cols_to_save = FEATURES + ["label"]
        
        # Garante que não faltam colunas
        for c in cols_to_save:
            if c not in df_dataset.columns:
                df_dataset[c] = 0.0
                
        df_dataset[cols_to_save].to_csv(DATASET_PATH, index=False)
        print(f"\n✅ BACKTEST CONCLUÍDO! Foram encontrados e gerados {len(df_dataset)} trades simulados.")
        print(f"📦 O seu mega-dataset foi guardado em: {DATASET_PATH}")
        
        # Pequeno resumo estatístico
        vitorias = df_dataset['label'].sum()
        print(f"📊 Win Rate Técnico Base (Sem ML): {(vitorias/len(df_dataset))*100:.2f}%")
    else:
        print("⚠ O backtest não encontrou sinais da estratégia.")

if __name__ == "__main__":
    generate_backtest_dataset()