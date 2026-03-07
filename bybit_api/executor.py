import time
import hmac
import hashlib
import requests
import json
from urllib.parse import urlencode
from config import API_KEY, API_SECRET, BASE_URL

def _generate_signature(params):
    query = urlencode(params)
    return hmac.new(
        API_SECRET.encode("utf-8"),
        query.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

def _headers():
    return {
        "X-BYBIT-API-KEY": API_KEY,
        "Content-Type": "application/json"
    }

def make_signed_request(method, endpoint, params={}):
    params['api_key'] = API_KEY
    params['timestamp'] = int(time.time() * 1000)
    params['sign'] = _generate_signature(params)

    url = f"{BASE_URL}{endpoint}"

    if method == "GET":
        response = requests.get(url, params=params, headers=_headers())
    else:
        response = requests.post(url, json=params, headers=_headers())

    try:
        return response.json()
    except Exception as e:
        print("Erro ao interpretar resposta:", e)
        return None

def get_account_balance():
    endpoint = "/v5/account/wallet-balance"
    params = {
        "accountType": "UNIFIED"
    }
    return make_signed_request("GET", endpoint, params)

def get_open_positions(symbol):
    endpoint = "/v5/position/list"
    params = {
        "category": "linear",
        "symbol": symbol
    }
    return make_signed_request("GET", endpoint, params)
