# trading/logger.py
from datetime import datetime

def log_event(event: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {event}"

    # Mostra no terminal
    print(log_line)

    # Salva no arquivo
    try:
        with open("trade_log.txt", "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
    except Exception as e:
        print(f"[ERRO] Falha ao salvar log: {e}")
