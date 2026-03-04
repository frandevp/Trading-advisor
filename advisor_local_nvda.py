import os
import time
import json
import requests
import numpy as np
import pandas as pd

from datetime import datetime, timezone
from dateutil.parser import isoparse

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient

from sklearn.ensemble import RandomForestRegressor


# =====================================================
# CONFIG
# =====================================================
TICKERS = ["NVDA", "IONQ", "QBTS", "CCJ", "QTUM", "RGTI"]
START_DATE = "2022-01-01"
CHECK_EVERY_SECONDS = 60 * 15

BUY_THRESHOLD = 0.002
SELL_THRESHOLD = -0.002

OLLAMA_MODEL = "llama3:latest"
OLLAMA_URL = "http://localhost:11434/api/generate"


# =====================================================
# FEATURES
# =====================================================
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_features(df):
    df = df.sort_index()

    df["return_1d"] = df["close"].pct_change()

    df["vol_10"] = df["return_1d"].rolling(10).std()
    df["vol_20"] = df["return_1d"].rolling(20).std()

    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()

    df["close_vs_sma20"] = (df["close"] / df["sma_20"]) - 1

    df["rsi_14"] = rsi(df["close"], 14)

    df["hl_range"] = (df["high"] / df["low"]) - 1
    df["co_range"] = (df["close"] / df["open"]) - 1

    df["log_volume"] = np.log(df["volume"].replace(0, np.nan))

    # lags
    df["ret_lag_1"] = df["return_1d"].shift(1)
    df["ret_lag_2"] = df["return_1d"].shift(2)
    df["ret_lag_3"] = df["return_1d"].shift(3)
    df["ret_lag_5"] = df["return_1d"].shift(5)

    df["vol20_lag_1"] = df["vol_20"].shift(1)
    df["rsi14_lag_1"] = df["rsi_14"].shift(1)

    return df


def make_target(df):
    df["target"] = df["close"].shift(-1) / df["close"] - 1
    return df


FEATURE_COLS = [
    "return_1d",
    "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_5",
    "vol_10", "vol_20", "vol20_lag_1",
    "sma_10", "sma_20", "sma_50",
    "close_vs_sma20",
    "rsi_14", "rsi14_lag_1",
    "hl_range", "co_range",
    "log_volume",
    "vwap",
]


# =====================================================
# ALPACA CLIENTS
# =====================================================
def get_keys():
    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Faltan API keys Alpaca (APCA_API_KEY_ID / APCA_API_SECRET_KEY).")
    return key, secret


def make_clients():
    key, secret = get_keys()
    data_client = StockHistoricalDataClient(api_key=key, secret_key=secret)
    trading_client = TradingClient(key, secret, paper=True)
    return data_client, trading_client


def fetch_bars(data_client, ticker: str):
    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Day,
        start=isoparse(START_DATE).replace(tzinfo=timezone.utc),
        end=datetime.now(timezone.utc),
        feed="iex",
    )

    bars = data_client.get_stock_bars(req).df

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(ticker)

    bars.index = pd.to_datetime(bars.index)
    return bars.sort_index()


def get_position_qty(trading_client, ticker: str) -> float:
    try:
        pos = trading_client.get_open_position(ticker)
        return float(pos.qty)
    except Exception:
        return 0.0


# =====================================================
# MODEL
# =====================================================
def train_and_predict(df):
    df = add_features(df)
    df = make_target(df)

    model_df = df[FEATURE_COLS + ["target", "close"]].dropna()

    split = int(len(model_df) * 0.8)
    train = model_df.iloc[:split]
    test = model_df.iloc[split:]

    X_train = train[FEATURE_COLS]
    y_train = train["target"]

    X_test = test[FEATURE_COLS]
    y_test = test["target"]

    model = RandomForestRegressor(
        n_estimators=600,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    last_X = model_df.iloc[[-1]][FEATURE_COLS]
    pred_next = float(model.predict(last_X)[0])

    preds = model.predict(X_test)
    mae = float(np.mean(np.abs(preds - y_test)))

    last_close = float(model_df.iloc[-1]["close"])

    return {
        "pred_next_return": pred_next,
        "last_close": last_close,
        "implied_next_close": last_close * (1 + pred_next),
        "mae_test": mae,
        "asof": str(model_df.index[-1]),
        "n_train": len(train),
        "n_test": len(test),
    }


def rule_action(pred):
    if pred >= BUY_THRESHOLD:
        return "BUY"
    if pred <= SELL_THRESHOLD:
        return "SELL"
    return "HOLD"


# =====================================================
# OLLAMA
# =====================================================
def ask_ollama(snapshot):
    prompt = f"""
Eres un asistente educativo de trading.

REGLAS:
- Solo puedes usar los valores del JSON.
- No inventes noticias, macroeconomía ni contexto externo.
- No prometas ganancias.
- No uses puntos suspensivos (...). Completa siempre todos los campos.

FORMATO OBLIGATORIO:

RECOMENDACIÓN: BUY/SELL/HOLD
JUSTIFICACIÓN: 2-3 frases usando pred_next_return, BUY_THRESHOLD, SELL_THRESHOLD y last_close.
SEÑAL: débil o fuerte (débil si abs(pred_next_return) < 0.001)
RIESGOS:
- Uno debe mencionar mae_test y que mide el error del modelo.
- Otro debe mencionar si la señal es débil o cercana a cero.
NO ACTUAR SI: una frase basada SOLO en HOLD, señal débil o mae_test alto.

JSON:
{json.dumps(snapshot, indent=2)}

BUY_THRESHOLD = {BUY_THRESHOLD}
SELL_THRESHOLD = {SELL_THRESHOLD}
"""

    r = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=90,
    )
    r.raise_for_status()
    return r.json()["response"]


# =====================================================
# MAIN
# =====================================================
def main():
    print("Advisor local MULTI iniciado (NO ejecuta órdenes). Ctrl+C para parar.")
    data_client, trading_client = make_clients()

    while True:
        print("\n" + "=" * 90)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Ejecutando ciclo para: {', '.join(TICKERS)}")

        for ticker in TICKERS:
            try:
                bars = fetch_bars(data_client, ticker)
                snap = train_and_predict(bars)
                qty = get_position_qty(trading_client, ticker)
                action = rule_action(snap["pred_next_return"])

                snapshot = {
                    "ticker": ticker,
                    "position_qty": qty,
                    "rule_action": action,
                    **snap,
                }

                print("\n" + "-" * 90)
                print(json.dumps(snapshot, indent=2))

                advice = ask_ollama(snapshot)
                print("\n--- ADVISOR (OLLAMA LOCAL) ---\n")
                print(advice)

            except Exception as e:
                print("\n" + "-" * 90)
                print(f"[{ticker}] [ERROR] {str(e)}")

        print("\n" + "=" * 90)
        time.sleep(CHECK_EVERY_SECONDS)


if __name__ == "__main__":
    main()