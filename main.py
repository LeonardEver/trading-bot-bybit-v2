import time
import csv
import joblib
import pandas as pd
import uuid
import os
from datetime import datetime, timedelta
from pathlib import Path
from utils.ohlcv import get_ohlcv
from strategies.strategy import generate_trade_signal
from sentiment.sentiment_analysis import get_news_sentiment
from trading.bybit_api import place_order, get_last_price, get_all_positions, close_position
from trading.risk_management import calculate_order_qty
from trading.logger import log_event
from utils.technical_indicators import calculate_indicators
from database.mongo_logger import log_trade, log_signal_decision, update_signal_outcome
from ml.config import FEATURES
from pybit.unified_trading import WebSocket


print(">>> Script main.py carregado com sucesso")


SYMBOL = "BTCUSDT"
LOOP_INTERVAL = 15
LOG_FILE = "trading_log.csv"

TAKE_PROFIT_PCT = 0.0010
STOP_LOSS_PCT = 0.0010
TRAILING_STOP_PCT = 0.0008

ultima_ordem = {"side": None, "hora": datetime.min}

# Variavel para Circuit Breaker
bloqueio_ate = datetime.min

# Pesos iniciais
peso_tecnico = 0.5
peso_sentimento = 0.5

# Caminho e features do modelo ML
MODEL_PATH = Path("ml/model_lgb.pkl")

# Carrega modelo, se existir
model = None
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
    log_event(f"Modelo ML carregado de {MODEL_PATH}")
else:
    log_event("⚠ Modelo ML não encontrado — rodando sem filtro de ML.")


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


def calcular_performance():
    """Ajusta pesos de decisão com base no PnL real (Dinheiro) gerado nos últimos trades."""
    global peso_tecnico, peso_sentimento

    try:
        if not Path(LOG_FILE).exists():
            return

        with open(LOG_FILE, 'r') as f:
            reader = list(csv.DictReader(f))
            
        if not reader:
            return

        ultimos = reader[-50:]
        if len(ultimos) < 10: # Não ajusta pesos se não tiver pelo menos 10 trades de histórico
            return

        pnl_tecnico_total = 0.0
        pnl_sentimento_total = 0.0

        for trade in ultimos:
            try:
                pnl = float(trade.get("pnl", 0))
                origem = trade.get("decision_source", "misto")

                if origem == "tecnico":
                    pnl_tecnico_total += pnl
                elif origem == "sentimento":
                    pnl_sentimento_total += pnl
                elif origem == "misto":
                    # Se foi misto, divide o mérito ou a culpa financeiramente
                    pnl_tecnico_total += pnl / 2
                    pnl_sentimento_total += pnl / 2
            except ValueError:
                continue

        # Evita divisão por zero e adiciona um "piso" de segurança
        if pnl_tecnico_total == 0 and pnl_sentimento_total == 0:
            return

        # --- A MÁGICA DA ADAPTAÇÃO POR RENTABILIDADE ---
        # Se os dois estão dando prejuízo, não mudamos os pesos agressivamente
        if pnl_tecnico_total <= 0 and pnl_sentimento_total <= 0:
            log_event("[ADAPTAÇÃO] Ambos os modelos em Drawdown. Mantendo pesos atuais de segurança.")
            return

        # Zera perdas para o cálculo de proporção (quem perde não ganha peso)
        peso_t_calc = max(0.01, pnl_tecnico_total)
        peso_s_calc = max(0.01, pnl_sentimento_total)

        soma = peso_t_calc + peso_s_calc
        
        # O novo peso alvo é estritamente proporcional a quem está ganhando mais dinheiro
        alvo_tecnico = peso_t_calc / soma
        alvo_sentimento = peso_s_calc / soma

        # Suavização: O bot não muda o peso do zero a 100% de uma vez. 
        # Ele caminha 10% em direção ao alvo a cada ciclo para não ser enganado por um único trade de sorte
        taxa_suavizacao = 0.10 
        peso_tecnico = (peso_tecnico * (1 - taxa_suavizacao)) + (alvo_tecnico * taxa_suavizacao)
        peso_sentimento = (peso_sentimento * (1 - taxa_suavizacao)) + (alvo_sentimento * taxa_suavizacao)

        # Garante limites máximos e mínimos
        peso_tecnico = max(0.1, min(0.9, peso_tecnico))
        peso_sentimento = 1 - peso_tecnico

        log_event(f"[ADAPTAÇÃO-PnL] Pesos Ajustados → Técnico: {peso_tecnico:.2f} | Sentimento: {peso_sentimento:.2f} | "
                  f"PnL Gerado: Téc ${pnl_tecnico_total:.2f}, Sent ${pnl_sentimento_total:.2f}")

    except Exception as e:
        log_event(f"[ERRO] Calcular performance: {e}")


def model_predict_prob(row):
    """Recebe última linha do DF e calcula probabilidade do label=1, atualizando o modelo dinamicamente"""
    global model, ultima_modificacao_modelo
    
    # Sistema de Hot-Reload: Verifica se o arquivo .pkl foi atualizado no disco
    if MODEL_PATH.exists():
        modificacao_atual = os.path.getmtime(MODEL_PATH)
        if modificacao_atual > ultima_modificacao_modelo:
            model = joblib.load(MODEL_PATH)
            ultima_modificacao_modelo = modificacao_atual
            log_event("🧠 [HOT-RELOAD] Novo modelo ML detectado e carregado em tempo de execução!")
            
    if model is None:
        return None
        
    try:
        df_row = pd.DataFrame([row])
        for f in FEATURES:
            if f not in df_row.columns:
                df_row[f] = 0.0  # preenche faltantes
        
        X = df_row[FEATURES].fillna(0)
        
        if hasattr(model, "predict_proba"):
            probabilidade = model.predict_proba(X)[0][1]
        else:
            probabilidade = model.predict(X)[0]
            
        return float(probabilidade)
        
    except Exception as e:
        log_event(f"[ERRO] Previsão ML: {e}")
        return None

def abrir_ordem():
    global ultima_ordem, bloqueio_ate # EDITADO: Adicionada global

        # ADICIONADO: Trava do Circuit Breaker para evitar perder mais dinheiro
    if datetime.now() < bloqueio_ate:
        log_event(f"⏳ Bot em modo Circuit Breaker. Operações suspensas até {bloqueio_ate.strftime('%H:%M:%S')}.")
        return

    # --- 1) Coleta dados de mercado ---
    df = get_ohlcv(SYMBOL)
    if df.empty:
        return

    df = calculate_indicators(df)

    #Log com DataFrame
    log_event(f"📊 Preço de fechamento real (DataFrame): {df['close'].iloc[-1]}")

    # --- 2) Análise de sentimento ---
    sentimento_str, sent_score = get_news_sentiment("BTC")
    if sentimento_str == "bullish":
        confiança_sentimento = 100
    elif sentimento_str == "bearish":
        confiança_sentimento = 0
    else:
        confiança_sentimento = 50

    df["sentiment_score"] = sent_score

    # Extrai hora/minuto
    ts = pd.to_datetime(df["timestamp"]) if "timestamp" in df.columns else pd.to_datetime(df.index)
    if isinstance(ts, pd.DatetimeIndex):
        df["hour"], df["minute"] = ts.hour, ts.minute
    else:
        df["hour"], df["minute"] = ts.dt.hour, ts.dt.minute

    log_event(f"Sentimento: {sentimento_str} ({confiança_sentimento}%)")

    # --- 3) Sinal técnico ---
    trade_decision = generate_trade_signal(df)
    sinal_tecnico = trade_decision.get("signal")
    confiança_tecnica = trade_decision.get("confidence", 50.0)

    confiança_final = (peso_tecnico * confiança_tecnica) + (peso_sentimento * confiança_sentimento)

    # Define risco
    if confiança_final >= 80:
        risk_level = "baixo"
    elif confiança_final >= 60:
        risk_level = "medio"
    else:
        risk_level = "alto"

    log_event(f"Sinal técnico: {sinal_tecnico} | Conf. técnica: {confiança_tecnica}% | "
              f"Conf. final: {confiança_final:.1f}% | Risco: {risk_level}")

    # --- 4) Machine Learning ---
    prob_ml = model_predict_prob(df.iloc[-1].to_dict())
    if prob_ml is not None:
        log_event(f"Probabilidade ML: {prob_ml:.3f}")
        if prob_ml < 0.6:
            log_event("⚠ ML filtrou a entrada — probabilidade abaixo do threshold.")
            return

        # ADICIONADO: Filtro de Consenso Mínimo (Quebra da ditadura do ML)
        if prob_ml > 0.60:
            if confiança_final < 50.0 and prob_ml < 0.85:
                log_event("⚠ Trade cancelado: ML quer operar contra a tendência, mas sem confluência macro/técnica.")
                return

    # --- DEBUG EXTRA (para investigar decisões) ---
    ultimo = df.iloc[-1]
    print(f"[DEBUG] ML prob: {prob_ml:.2f} | Sentimento: {sentimento_str} ({sent_score:.2f})")
    print(f"[DEBUG] RSI: {ultimo['rsi']:.2f} | MACD: {ultimo['macd']:.2f} | "
          f"Signal: {ultimo['macd_signal']:.2f} | EMA20: {ultimo['ema_20']:.2f} | EMA50: {ultimo['ema_50']:.2f}")
    print(f"[DEBUG] Decisão: {'Abrir trade' if sinal_tecnico in ['buy', 'sell'] else 'Ignorar sinal'}")

    # --- 5) Valida sinal ---
    if sinal_tecnico not in ["buy", "sell"]:
        return

    posicoes = get_all_positions(SYMBOL) or []
    if (ultima_ordem.get("side") == sinal_tecnico and
        datetime.now() - ultima_ordem.get("hora", datetime.min) < timedelta(seconds=10) and
        len(posicoes) == 0):
        print("⚠ Ordem recente no mesmo sentido (cooldown). Ignorando.")
        return

    for pos in posicoes:
        if pos.get("side", "").lower() == sinal_tecnico:
            print(f"⚠ Já existe posição {sinal_tecnico.upper()} aberta. Ordem ignorada.")
            return

    price = get_last_price(SYMBOL)
    if price is None:
        return

    qty = calculate_order_qty(SYMBOL, risk_level, price)

    # Obter o ATR do candle mais recente
    atr_atual = df.iloc[-1]['atr']

    # Multiplicadores profissionais
    # Stop Loss a 1.5x o ATR protege contra ruídos de violinada
    # Take Profit a 2.0x o ATR garante um Risco/Retorno positivo
    SL_MULTIPLIER = 1.5
    TP_MULTIPLIER = 2.0

    # --- 6) Calcula TP / SL Dinâmicos ---
    if sinal_tecnico == "buy":
        side = "Buy"  # <--- ADICIONE ESTA LINHA
        take_profit = round(price + (atr_atual * TP_MULTIPLIER), 2)
        stop_loss = round(price - (atr_atual * SL_MULTIPLIER), 2)
    else: # "sell"
        side = "Sell" # <--- ADICIONE ESTA LINHA
        take_profit = round(price - (atr_atual * TP_MULTIPLIER), 2)
        stop_loss = round(price + (atr_atual * SL_MULTIPLIER), 2)

    # Trailing stop pode acompanhar a volatilidade também (ex: metade do ATR)
    trailing_stop = round((atr_atual * 0.5), 2)

    # --- 7) Envia ordem ---
    log_event(f"Abrindo {side} | Preço: {price} | TP: {take_profit} | SL: {stop_loss} | TS: {trailing_stop}")
    order_result = place_order(
        SYMBOL, side, qty,
        str(take_profit),
        str(stop_loss),
        str(trailing_stop)
    )

    ok = False
    if isinstance(order_result, dict):
        ok = (order_result.get("retCode") == 0) or (order_result.get("success") is True) or bool(order_result.get("result"))
    else:
        ok = bool(order_result)

    if not ok:
        log_event(f"❌ Falha ao abrir ordem {side} em {SYMBOL}: {order_result}")
        return

    # --- 8) trade_id + log de decisão ---
    trade_id = str(uuid.uuid4())
    decision_doc = {
        "trade_id": trade_id,
        "timestamp": datetime.now(),
        "symbol": SYMBOL,
        "side": side,
        "entry_price": float(price),
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
    log_signal_decision(decision_doc)


    # --- 9) Atualiza ultima_ordem (inclui trade_id e alvos) ---
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
        "entry_price": float(price),
        "qty": float(qty),
    }

def fechar_ordem(side, entry_price, size, current_price):
    # Fecha posição real na corretora
    close_position(SYMBOL, side)

    # Calcula PnL e % com Taxas da Corretora embutidas (0.05% entrada + 0.05% saída)
    taxa_corretora = 0.0005
    custo_taxas = (entry_price * size * taxa_corretora) + (current_price * size * taxa_corretora)
    
    pnl_bruto = (current_price - entry_price) * size if side == "Buy" else (entry_price - current_price) * size
    pnl = pnl_bruto - custo_taxas
    
    pnl_pct = ((current_price / entry_price) - 1) * 100 if side == "Buy" else ((entry_price / current_price) - 1) * 100

    # Monta trade_data para logs
    trade_data = {
        "timestamp": datetime.now(),
        "trade_id": ultima_ordem.get("trade_id"),
        "symbol": SYMBOL,
        "side": side,
        "entry_price": float(entry_price),
        "exit_price": float(current_price),
        "qty": float(size),
        "pnl": round(float(pnl), 2),
        "pnl_pct": round(float(pnl_pct), 3),
        "decision_source": ultima_ordem.get("origem", "misto"),
        "risk_level": ultima_ordem.get("risk_level", "desconhecido"),
        "take_profit": ultima_ordem.get("take_profit"),
        "stop_loss": ultima_ordem.get("stop_loss"),
        "trailing_stop": ultima_ordem.get("trailing_stop"),
        "sentiment_str": ultima_ordem.get("sentiment_str"),
        "sentiment_score": ultima_ordem.get("sentiment_score"),
        "ml_probability": ultima_ordem.get("ml_probability"),
        "ema_20": ultima_ordem.get("ema_20"),
        "ema_50": ultima_ordem.get("ema_50"),
        "ema_200": ultima_ordem.get("ema_200"),
        "rsi": ultima_ordem.get("rsi"),
        "macd": ultima_ordem.get("macd"),
        "macd_signal": ultima_ordem.get("macd_signal"),
        "macd_hist": ultima_ordem.get("macd_hist"),
        "bb_width": ultima_ordem.get("bb_width"),
        "atr": ultima_ordem.get("atr"),
        "volume": ultima_ordem.get("volume"),
        "volume_ma": ultima_ordem.get("volume_ma"),
        "hour": ultima_ordem.get("hour"),
        "minute": ultima_ordem.get("minute")
    }

    # Salva em CSV e log local
    salvar_log_csv(trade_data)
    log_event(f"{side} fechado - PnL: {pnl:.2f} ({pnl_pct:.3f}%) | Origem: {trade_data['decision_source']}")
    log_trade(trade_data)

    # Atualiza outcome no Mongo (ml_dataset)
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

    # Reseta cooldown para liberar novas ordens
    ultima_ordem["side"] = None
    ultima_ordem["hora"] = datetime.min

def monitorar_posicoes():
    posicoes = get_all_positions(SYMBOL)
    if not posicoes:
        return

    for pos in posicoes:
        entry_price = float(pos.get("avgPrice", 0))
        side = pos.get("side")
        size = float(pos.get("size", 0))
        current_price = get_last_price(SYMBOL)

        if not current_price:
            continue

        print(f"[MONITOR] {side} {size} @ {entry_price} | Preço atual: {current_price}")

        fechar = False
        tp_real = ultima_ordem.get("take_profit", entry_price * 1.02)
        sl_real = ultima_ordem.get("stop_loss", entry_price * 0.98)

        if side == "Buy":
            if current_price >= tp_real or current_price <= sl_real:
                fechar = True
        elif side == "Sell":
            if current_price <= tp_real or current_price >= sl_real:
                fechar = True
                
        if fechar:
                close_position(SYMBOL, side)

                # --- INÍCIO DA MATEMÁTICA DE TAXAS (CORREÇÃO 3) ---
                taxa_corretora = 0.0005 # 0.05% de taxa da Bybit
                custo_taxas = (entry_price * size * taxa_corretora) + (current_price * size * taxa_corretora)

                pnl_bruto = (current_price - entry_price) * size if side == "Buy" else (entry_price - current_price) * size
                pnl = pnl_bruto - custo_taxas
                
                pnl_pct = ((current_price / entry_price) - 1) * 100 if side == "Buy" else ((entry_price / current_price) - 1) * 100
                # --- FIM DA MATEMÁTICA DE TAXAS ---

                trade_data = {
                    "timestamp": datetime.now(),
                    "trade_id": ultima_ordem.get("trade_id"),   # <--- vincula entrada/saída
                    "symbol": SYMBOL,
                    "side": side,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "qty": size,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 3),
                    "decision_source": ultima_ordem.get("origem", "misto"),
                    "risk_level": ultima_ordem.get("risk_level", "desconhecido"),
                    "take_profit": ultima_ordem.get("take_profit"),
                    "stop_loss": ultima_ordem.get("stop_loss"),
                    "sentiment_str": ultima_ordem.get("sentiment_str"),
                    "sentiment_score": ultima_ordem.get("sentiment_score"),
                    "ml_probability": ultima_ordem.get("ml_probability"),
                    "ema_20": ultima_ordem.get("ema_20"),
                    "ema_50": ultima_ordem.get("ema_50"),
                    "ema_200": ultima_ordem.get("ema_200"),
                    "rsi": ultima_ordem.get("rsi"),
                    "macd": ultima_ordem.get("macd"),
                    "macd_signal": ultima_ordem.get("macd_signal"),
                    "macd_hist": ultima_ordem.get("macd_hist"),
                    "bb_width": ultima_ordem.get("bb_width"),
                    "atr": ultima_ordem.get("atr"),
                    "volume": ultima_ordem.get("volume"),
                    "volume_ma": ultima_ordem.get("volume_ma"),
                    "hour": ultima_ordem.get("hour"),
                    "minute": ultima_ordem.get("minute")
                }

                salvar_log_csv(trade_data)
                log_event(f"{side} fechado - PnL: {pnl:.2f} ({pnl_pct:.3f}%) | Origem: {trade_data['decision_source']}")
                log_trade(trade_data)

                # >>> Atualiza outcome no documento de decisão do ML
                trade_id = trade_data.get("trade_id")
                if trade_id:
                    label = 1 if pnl > 0 else 0
                    update_signal_outcome(trade_id, {
                        "exit_price": float(current_price),
                        "pnl": float(round(pnl, 2)),
                        "pnl_pct": float(round(pnl_pct, 3)),
                        "label": int(label)
                    })


# =======================================================
if __name__ == "__main__":
    from pybit.unified_trading import WebSocket
    import threading
    
    log_event("🚀 Bot iniciado. Configurando WebSockets...")

    def handle_kline(message):
        """Callback acionado pela Bybit"""
        data = message.get("data", [])
        if data:
            candle = data[0]
            if candle.get("confirm"):
                # O preço será logado de forma correta e garantida lá na função abrir_ordem()
                log_event("🕯️ Candle fechado! Iniciando extração de dados da Corretora e análise de ML...")
                
                # --- A MÁGICA DO MULTI-THREADING AQUI ---
                # Criamos uma função interna só para encapsular o trabalho pesado
                def executar_trabalho_pesado():
                    try:
                        calcular_performance()
                        abrir_ordem()
                    except Exception as e:
                        log_event(f"❌ Erro durante a análise pós-candle: {e}")

                # Disparamos o trabalho pesado em uma nova Thread (em paralelo)
                # Assim, a função handle_kline termina instantaneamente e o WebSocket não trava!
                thread_analise = threading.Thread(target=executar_trabalho_pesado)
                thread_analise.start()
                # ----------------------------------------

    # 1. Inicia a conexão WebSocket
    ws = WebSocket(
        testnet=True, 
        channel_type="linear"
    )
    
    # 2. Assina o tempo gráfico
    ws.kline_stream(
        interval="15", 
        symbol=SYMBOL, 
        callback=handle_kline
    )
    
    log_event("📡 WebSocket conectado! Aguardando o fechamento do próximo candle.")

    while True:
        try:
            monitorar_posicoes()
        except Exception as e:
            log_event(f"❌ Erro ao monitorar posições: {e}")
            
        time.sleep(LOOP_INTERVAL)