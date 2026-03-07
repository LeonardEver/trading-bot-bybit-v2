# database/mongo_logger.py
from pymongo import MongoClient
from datetime import datetime
from trading.logger import log_event
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MONGO_DB", "trading_bot")

def _connect():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        client.server_info()
        db = client[DB_NAME]
        log_event(f"[MONGO] Conectado em {MONGO_URI} / DB={DB_NAME}")
        return db
    except Exception as e:
        log_event(f"[MONGO ERRO] Conexão falhou: {e}")
        return None

_db = _connect()
trades_collection = _db["trades"] if _db is not None else None
signals_collection = _db["signals"] if _db is not None else None
ml_dataset_collection = _db["ml_dataset"] if _db is not None else None

def _ensure():
    global _db, trades_collection, signals_collection, ml_dataset_collection
    if _db is None:
        _db = _connect()
        if _db:
            trades_collection = _db["trades"]
            signals_collection = _db["signals"]
            ml_dataset_collection = _db["ml_dataset"]
    return _db is not None

# ---------- Funções de logging ----------

def log_trade(data):
    """Insere trade fechado em 'trades'."""
    if not _ensure():
        log_event("[MONGO] Conexão indisponível — trade não salvo.")
        return False
    try:
        data.setdefault("timestamp", datetime.now())
        trades_collection.insert_one(data)
        log_event(f"[MONGO] Trade salvo: {data.get('symbol')} {data.get('side')} | PnL: {data.get('pnl')}")
        return True
    except Exception as e:
        log_event(f"[MONGO ERRO] Falha ao salvar trade: {e}")
        return False


def log_signal_decision(doc):
    """
    Insere decisão de entrada com features/scores em 'ml_dataset'.
    Retorna o trade_id para vincular depois no fechamento.
    """
    if not _ensure():
        log_event("[MONGO] Conexão indisponível — decisão não salva.")
        return None
    try:
        doc = dict(doc)
        doc.setdefault("created_at", datetime.now())
        doc.setdefault("status", "open")  # open -> closed
        ml_dataset_collection.insert_one(doc)
        log_event(f"[MONGO] Decisão registrada em ml_dataset. trade_id={doc.get('trade_id')}")
        return doc.get("trade_id")
    except Exception as e:
        log_event(f"[MONGO ERRO] Falha ao salvar decisão ML: {e}")
        return None


def update_signal_outcome(trade_id, outcome):
    """
    Atualiza documento em 'ml_dataset' com resultado (PnL/label).
    outcome: dict com exit_price, pnl, pnl_pct, label, closed_at, etc.
    """
    if not _ensure():
        log_event("[MONGO] Conexão indisponível — outcome não salvo.")
        return False
    try:
        outcome = dict(outcome)
        outcome.setdefault("closed_at", datetime.now())
        outcome["status"] = "closed"
        res = ml_dataset_collection.update_one({"trade_id": trade_id}, {"$set": outcome})
        if res.matched_count == 0:
            log_event(f"[MONGO] Nenhum ml_dataset com trade_id={trade_id} para atualizar.")
            return False
        log_event(f"[MONGO] Outcome atualizado em ml_dataset. trade_id={trade_id}")
        return True
    except Exception as e:
        log_event(f"[MONGO ERRO] Falha ao atualizar outcome: {e}")
        return False
