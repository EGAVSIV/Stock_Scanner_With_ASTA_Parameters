import os
import pandas as pd
import numpy as np
import streamlit as st
import talib

# ==========================
# CONFIG
# ==========================
st.set_page_config("Multi TF Stock RUN Scanner", layout="wide")
st.title("🚀 Multi-Timeframe Stock RUN Analyzer")

DATA_1H = "stock_data_1H"
DATA_D = "stock_data_D"
DATA_W = "stock_data_W"

# ==========================
# DATA LOADER
# ==========================
@st.cache_data
def load_stock(folder, symbol):
    path = os.path.join(folder, f"{symbol}.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df = df.sort_values("Datetime")
    return df.reset_index(drop=True)

# ==========================
# BEST MA SELECTOR
# ==========================
def best_ma(df):
    df["EMA20"] = talib.EMA(df["Close"], 20)
    df["EMA50"] = talib.EMA(df["Close"], 50)
    df["SMA20"] = talib.SMA(df["Close"], 20)
    df["SMA50"] = talib.SMA(df["Close"], 50)

    slopes = {
        "EMA20": df["EMA20"].diff().iloc[-1],
        "EMA50": df["EMA50"].diff().iloc[-1],
        "SMA20": df["SMA20"].diff().iloc[-1],
        "SMA50": df["SMA50"].diff().iloc[-1],
    }
    return max(slopes, key=slopes.get)

# ==========================
# HOURLY RUN DETECTION
# ==========================
def hourly_run(df):
    close = df["Close"]
    pct_move = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100

    macd, signal, hist = talib.MACD(close)
    rsi = talib.RSI(close, 14)
    adx = talib.ADX(df["High"], df["Low"], close, 14)

    return (
        pct_move >= 4
        and hist.iloc[-1] > hist.iloc[-2]
        and rsi.iloc[-1] > 55
        and adx.iloc[-1] > adx.iloc[-2]
    ), pct_move

# ==========================
# DAILY CONFIRMATION
# ==========================
def daily_confirm(df):
    ma = best_ma(df)
    df["MA"] = df[ma]
    rsi = talib.RSI(df["Close"], 14)
    return df["Close"].iloc[-1] > df["MA"].iloc[-1] and rsi.iloc[-1] > 50, ma

# ==========================
# WEEKLY CONFIRMATION
# ==========================
def weekly_confirm(df):
    ema20 = talib.EMA(df["Close"], 20)
    ema50 = talib.EMA(df["Close"], 50)
    adx = talib.ADX(df["High"], df["Low"], df["Close"], 14)
    return ema20.iloc[-1] > ema50.iloc[-1] and adx.iloc[-1] > 20

# ==========================
# STOCK LIST
# ==========================
symbols = [f.replace(".parquet", "") for f in os.listdir(DATA_1H)]
selected = st.multiselect("📌 Select Stocks", symbols, default=symbols[:10])

results = []

# ==========================
# MAIN LOOP
# ==========================
for sym in selected:
    df1h = load_stock(DATA_1H, sym)
    dfd = load_stock(DATA_D, sym)
    dfw = load_stock(DATA_W, sym)

    if df1h is None or dfd is None or dfw is None:
        continue

    run, move = hourly_run(df1h)
    dconf, ma_used = daily_confirm(dfd)
    wconf = weekly_confirm(dfw)

    if run and dconf and wconf:
        results.append({
            "Stock": sym,
            "Hourly Move %": round(move, 2),
            "Daily MA Used": ma_used,
            "Daily Trend": "✔",
            "Weekly Trend": "✔",
            "RUN Reason": "Hourly Momentum + HTF Alignment"
        })

# ==========================
# OUTPUT
# ==========================
if results:
    df_out = pd.DataFrame(results)
    st.success(f"🔥 {len(df_out)} Stocks in RUN Mode")
    st.dataframe(df_out, use_container_width=True)
else:
    st.warning("No RUN candidates found")
