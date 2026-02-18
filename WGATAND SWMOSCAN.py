import streamlit as st
import pandas as pd
import talib as ta
import os
from pathlib import Path

st.set_page_config(page_title="Master Stock Scanner", layout="wide")

st.title("📈 Multi-Timeframe Master Scanner")

# ===============================
# DATA FOLDERS
# ===============================
DATA_D = "stock_data_D"
DATA_W = "stock_data_W"
DATA_M = "stock_data_M"

symbols = list(set(
    [f.stem for f in Path(DATA_D).glob("*.parquet")]
))

st.write(f"✅ Found {len(symbols)} symbols")

# ===============================
# FUNCTIONS
# ===============================

def get_macd_trend(series):
    macd, signal, hist = ta.MACD(series, 12, 26, 9)
    if macd.dropna().shape[0] < 3:
        return None, None
    trend_now = "Up Tick" if macd.iloc[-1] > macd.iloc[-2] else "Down Tick"
    trend_prev = "Up Tick" if macd.iloc[-2] > macd.iloc[-3] else "Down Tick"
    return trend_now, trend_prev


def classify_trend(daily, daily_n1, weekly, monthly):

    if daily == "Up Tick" and weekly == "Up Tick" and monthly == "Up Tick":
        if daily_n1 == "Up Tick":
            return "Running Uptrend"
        else:
            return "D Aligned Up With W/M"

    if daily == "Down Tick" and weekly == "Down Tick" and monthly == "Down Tick":
        if daily_n1 == "Down Tick":
            return "Running Downtrend"
        else:
            return "D Aligned Down With W/M"

    if daily == "Down Tick" and weekly == "Up Tick" and monthly == "Up Tick":
        return "Wave Going Against Tide (Down)"

    if daily == "Up Tick" and weekly == "Down Tick" and monthly == "Down Tick":
        return "Wave Going Against Tide (Up)"

    return "No Clear Trend"


# ===============================
# SCAN BUTTON
# ===============================
if st.button("🚀 Run Full Scan"):

    results = []

    for symbol in symbols:
        try:
            df_d = pd.read_parquet(f"{DATA_D}/{symbol}.parquet")
            df_w = pd.read_parquet(f"{DATA_W}/{symbol}.parquet")
            df_m = pd.read_parquet(f"{DATA_M}/{symbol}.parquet")

            if len(df_d) < 100:
                continue

            # ===============================
            # MTF MACD
            # ===============================
            d_now, d_prev = get_macd_trend(df_d["close"])
            w_now, _ = get_macd_trend(df_w["close"])
            m_now, _ = get_macd_trend(df_m["close"])

            if not d_now or not w_now or not m_now:
                continue

            trend_status = classify_trend(d_now, d_prev, w_now, m_now)

            # ===============================
            # DAILY MOMENTUM STRUCTURE
            # ===============================
            df_d["ema13"] = ta.EMA(df_d["close"], 13)
            df_d["ema50"] = ta.EMA(df_d["close"], 50)
            df_d["ema100"] = ta.EMA(df_d["close"], 100)
            df_d["rsi"] = ta.RSI(df_d["close"], 14)
            df_d["adx"] = ta.ADX(df_d["high"], df_d["low"], df_d["close"], 14)

            latest = df_d.iloc[-1]

            bullish_momentum = (
                latest["ema13"] > latest["ema50"] > latest["ema100"]
                and latest["adx"] > 20
                and latest["rsi"] > 55
                and latest["close"] > latest["ema13"]
            )

            bearish_momentum = (
                latest["ema13"] < latest["ema50"] < latest["ema100"]
                and latest["adx"] > 20
                and latest["rsi"] < 45
                and latest["close"] < latest["ema13"]
            )

            bullish_swing = (
                latest["ema13"] > latest["ema50"]
                and latest["rsi"] < 55
            )

            bearish_swing = (
                latest["ema13"] < latest["ema50"]
                and latest["rsi"] > 45
            )

            results.append({
                "Stock": symbol,
                "Trend Status": trend_status,
                "Bullish Momentum": bullish_momentum,
                "Bearish Momentum": bearish_momentum,
                "Bullish Swing": bullish_swing,
                "Bearish Swing": bearish_swing
            })

        except Exception as e:
            continue

    df_result = pd.DataFrame(results)

    st.success("Scan Completed ✅")

    st.dataframe(df_result, use_container_width=True)

    # ===============================
    # CATEGORY FILTERS
    # ===============================
    st.subheader("📊 Category View")

    category = st.selectbox(
        "Select Category",
        [
            "All",
            "Running Uptrend",
            "Running Downtrend",
            "Wave Going Against Tide (Down)",
            "Wave Going Against Tide (Up)",
            "No Clear Trend"
        ]
    )

    if category != "All":
        filtered = df_result[df_result["Trend Status"] == category]
        st.dataframe(filtered, use_container_width=True)
