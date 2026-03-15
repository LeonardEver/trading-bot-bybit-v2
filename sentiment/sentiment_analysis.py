# sentiment/sentiment_analysis.py

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# Variáveis Globais de Cache (Memória de 12h)
_CACHE_SENTIMENT_STR = "neutral"
_CACHE_SENTIMENT_SCORE = 0.0
_CACHE_TIMESTAMP = 0
CACHE_DURATION_SECONDS = 12 * 3600  # 12 horas

def get_news_sentiment(symbol="BTC"):
    """
    Busca o Índice de Medo e Ganância via API da CoinMarketCap.
    Retorna uma tupla: (sentiment_str, sentiment_score_normalizado)
    """
    global _CACHE_SENTIMENT_STR, _CACHE_SENTIMENT_SCORE, _CACHE_TIMESTAMP
    
    agora = time.time()
    
    # Retorna o cache se ainda estiver dentro da validade
    if (agora - _CACHE_TIMESTAMP) < CACHE_DURATION_SECONDS:
        return _CACHE_SENTIMENT_STR, _CACHE_SENTIMENT_SCORE

    print("📰 [CACHE EXPIRADO] Atualizando Fear & Greed Index via CoinMarketCap...")
    
    api_key = os.getenv("CMC_API_KEY")
    if not api_key:
        print("⚠ [AVISO] 'CMC_API_KEY' não encontrada no .env.")
        return _CACHE_SENTIMENT_STR, _CACHE_SENTIMENT_SCORE

    url = "https://pro-api.coinmarketcap.com/v3/fear-and-greed/latest"
    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": api_key
    }

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        data = resp.json()
        
        # O SEGREDO ESTAVA AQUI: Convertendo para string para garantir a igualdade
        status_code = str(data.get("status", {}).get("error_code"))
        
        if status_code == "0":
            fg_value = data.get("data", {}).get("value")
            
            if fg_value is not None:
                fg_value = int(fg_value)
                
                # Normaliza de 0-100 para a escala do Machine Learning (-1.0 a 1.0)
                normalized_score = (fg_value - 50) / 50.0
                
                # Define a string classificatória
                if fg_value >= 55:
                    sentiment_str = "bullish"
                elif fg_value <= 45:
                    sentiment_str = "bearish"
                else:
                    sentiment_str = "neutral"
                    
                # Salva na memória
                _CACHE_SENTIMENT_STR = sentiment_str
                _CACHE_SENTIMENT_SCORE = round(normalized_score, 2)
                _CACHE_TIMESTAMP = agora
                
                print(f"✅ [SENTIMENTO CMC] Valor: {fg_value}/100 | Score ML: {_CACHE_SENTIMENT_SCORE} | Viés: {_CACHE_SENTIMENT_STR.upper()}")
            else:
                print("❌ [ERRO CMC] Valor 'value' não encontrado no nó 'data'.")
                
        else:
            error_msg = data.get("status", {}).get("error_message", "Erro desconhecido")
            print(f"❌ [ERRO CMC] Código {status_code}: {error_msg}")

    except Exception as e:
        print(f"❌ [ERRO DE REDE] Falha ao conectar com CMC: {e}")
        
    return _CACHE_SENTIMENT_STR, _CACHE_SENTIMENT_SCORE