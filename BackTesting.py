import os
import pandas as pd
import numpy as np
import streamlit as st
import talib

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(
    page_title="Stock Behavioral Intelligence Engine",
    layout="wide",
    page_icon="🧠"
)

st.title("🧠 Stock Behavioral Intelligence Engine")
st.caption("Behavioral backtesting → verdict → confidence → alert")

# =====================================================
# DATA PATHS
# =====================================================
DATA_D = "stock_data_D"
DATA_W = "stock_data_W"
DATA_M = "stock_data_M"

# =====================================================
# SAFE DATA LOADER
# =====================================================
def load_stock(folder, symbol):
    path = os.path.join(folder, f"{symbol}.parquet")
    if not os.path.exists(path):
        return None

    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]

    dt = next((c for c in ["date", "datetime", "timestamp"] if c in df.columns), None)
    if dt:
        df[dt] = pd.to_datetime(df[dt])
        df = df.sort_values(dt)

    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            return None

    return df.reset_index(drop=True)

# =====================================================
# EVENT DETECTION (4% in next 3 days)
# =====================================================
def detect_events(df):
    future_move = (df["close"].shift(-3) - df["close"]) / df["close"] * 100
    return df[future_move >= 4].index

# =====================================================
# INDICATOR SNAPSHOT
def snapshot(df, i):
    close = df["close"]

    rsi = talib.RSI(close, 14)
    adx = talib.ADX(df["high"], df["low"], close, 14)
    macd, signal, _ = talib.MACD(close)
    ema20 = talib.EMA(close, 20)
    bb_u, bb_m, bb_l = talib.BBANDS(close, 20)

    # Convert i to positional index safely
    if i < 0:
        i = len(df) + i

    # Guard against insufficient data
    if i < 0 or i >= len(df):
        return None

    if pd.isna(rsi.iloc[i]) or pd.isna(adx.iloc[i]):
        return None

    return {
        "RSI": float(rsi.iloc[i]),
        "ADX": float(adx.iloc[i]),
        "MACD": "Bullish" if macd.iloc[i] > signal.iloc[i] else "Bearish",
        "Price>EMA20": bool(close.iloc[i] > ema20.iloc[i]),
        "BB": (
            "Upper" if close.iloc[i] > bb_u.iloc[i]
            else "Lower" if close.iloc[i] < bb_l.iloc[i]
            else "Middle"
        )
    }


# =====================================================
# HIGHER TF SNAPSHOT (LATEST)
# =====================================================
def latest_tf_state(df, label):
    close = df["close"]
    rsi = talib.RSI(close, 14).iloc[-1]
    macd, signal, _ = talib.MACD(close)
    ema20 = talib.EMA(close, 20)
    ema50 = talib.EMA(close, 50)

    return {
        f"{label}_RSI": rsi,
        f"{label}_MACD": "Bullish" if macd.iloc[-1] > signal.iloc[-1] else "Bearish",
        f"{label}_Trend": "Up" if ema20.iloc[-1] > ema50.iloc[-1] else "Down"
    }

# =====================================================
# BUILD FINAL VERDICT
# =====================================================
def build_verdict(df):
    verdict = []

    if df["RSI"].mean() >= 60:
        verdict.append("RSI>60")
    if df["ADX"].mean() >= 25:
        verdict.append("ADX>25")
    if df["Price>EMA20"].mean() >= 0.65:
        verdict.append("Price>EMA20")
    if (df["MACD"] == "Bullish").mean() >= 0.65:
        verdict.append("Daily MACD Bullish")

    return verdict

# =====================================================
# CONFIDENCE SCORE
# =====================================================
def confidence_score(verdict, current):
    matched = 0

    for rule in verdict:
        if rule == "RSI>60" and current["Daily_RSI"] > 60:
            matched += 1
        elif rule == "ADX>25" and current["Daily_ADX"] > 25:
            matched += 1
        elif rule == "Price>EMA20" and current["Price>EMA20"]:
            matched += 1
        elif rule == "Daily MACD Bullish" and current["Daily_MACD"] == "Bullish":
            matched += 1

    return int((matched / len(verdict)) * 100) if verdict else 0

# =====================================================
# STOCK SELECTION
# =====================================================
symbols = sorted([f.replace(".parquet", "") for f in os.listdir(DATA_D)])
selected = st.multiselect("📌 Select Stocks", symbols, default=symbols[:5])

# =====================================================
# RUN ENGINE
# =====================================================
if st.button("🚀 Run Behavioral Intelligence"):
    alerts = []

    for sym in selected:
        df_d = load_stock(DATA_D, sym)
        df_w = load_stock(DATA_W, sym)
        df_m = load_stock(DATA_M, sym)

        if df_d is None or df_w is None or df_m is None:
            continue

        events = detect_events(df_d)
        if len(events) < 3:
            continue

        rows = []
        for i in events:
            s = snapshot(df_d, i)
            rows.append(s)

        hist_df = pd.DataFrame(rows)
        verdict = build_verdict(hist_df)

        # Current market snapshot
        latest = snapshot(df_d, len(df_d) - 1)
        if latest is None:
            continue
        latest["Daily_RSI"] = latest["RSI"]
        latest["Daily_ADX"] = latest["ADX"]
        latest["Daily_MACD"] = latest["MACD"]

        score = confidence_score(verdict, latest)

        if score >= 70:
            alerts.append({
                "Stock": sym,
                "Confidence %": score,
                "Verdict": ", ".join(verdict)
            })

    if alerts:
        st.success("🔥 High-Confidence Behavioral Matches")
        st.dataframe(pd.DataFrame(alerts), use_container_width=True)
    else:
        st.warning("No stocks with ≥70% behavioral alignment")
