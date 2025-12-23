import os
import pandas as pd
import numpy as np
import streamlit as st
import talib

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(
    page_title="Stock Personality Backtest Engine",
    layout="wide",
    page_icon="🧠"
)

st.title("🧠 Stock Personality Backtest Engine")
st.caption("Learn WHY stocks run – from historical behavior")

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

    dt_col = next((c for c in ["datetime", "date", "timestamp"] if c in df.columns), None)
    if dt_col:
        df[dt_col] = pd.to_datetime(df[dt_col])
        df = df.sort_values(dt_col)

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            return None

    return df.reset_index(drop=True)

# =====================================================
# EVENT DETECTION (4% in next 3 days)
# =====================================================
def detect_events(df):
    close = df["close"]
    future_move = (close.shift(-3) - close) / close * 100
    return df[future_move >= 4].index

# =====================================================
# DAILY CONDITION SNAPSHOT
# =====================================================
def daily_snapshot(df, i):
    close = df["close"]

    rsi = talib.RSI(close, 14)
    adx = talib.ADX(df["high"], df["low"], close, 14)
    macd, signal, _ = talib.MACD(close)

    ema20 = talib.EMA(close, 20)
    sma20 = talib.SMA(close, 20)
    bb_u, bb_m, bb_l = talib.BBANDS(close, 20)

    return {
        "Daily_RSI": round(rsi[i], 2),
        "Daily_ADX": round(adx[i], 2),
        "Daily_MACD": "Bullish" if macd[i] > signal[i] else "Bearish",
        "Price>EMA20": close[i] > ema20[i],
        "Price>SMA20": close[i] > sma20[i],
        "BB_Position":
            "Upper" if close[i] > bb_u[i] else
            "Lower" if close[i] < bb_l[i] else
            "Middle"
    }

# =====================================================
# WEEKLY & MONTHLY SNAPSHOT
# =====================================================
def higher_tf_snapshot(df, label):
    close = df["close"]

    rsi = talib.RSI(close, 14)
    adx = talib.ADX(df["high"], df["low"], close, 14)
    macd, signal, _ = talib.MACD(close)

    ema20 = talib.EMA(close, 20)
    ema50 = talib.EMA(close, 50)

    return {
        f"{label}_RSI": round(rsi.iloc[-1], 2),
        f"{label}_ADX": round(adx.iloc[-1], 2),
        f"{label}_MACD": "Bullish" if macd.iloc[-1] > signal.iloc[-1] else "Bearish",
        f"{label}_Trend": "Up" if ema20.iloc[-1] > ema50.iloc[-1] else "Down"
    }

# =====================================================
# STOCK SELECTION
# =====================================================
symbols = sorted([f.replace(".parquet", "") for f in os.listdir(DATA_D)])
selected = st.multiselect("📌 Select Stocks", symbols, default=symbols[:5])

# =====================================================
# RUN BACKTEST
# =====================================================
if st.button("🔍 Learn Stock Behavior"):
    all_events = []

    for sym in selected:
        df_d = load_stock(DATA_D, sym)
        df_w = load_stock(DATA_W, sym)
        df_m = load_stock(DATA_M, sym)

        if df_d is None or df_w is None or df_m is None:
            continue

        events = detect_events(df_d)

        for i in events:
            row = {
                "Stock": sym,
                "Event_Date": df_d.iloc[i]["date"] if "date" in df_d.columns else i
            }
            row.update(daily_snapshot(df_d, i))
            row.update(higher_tf_snapshot(df_w, "Weekly"))
            row.update(higher_tf_snapshot(df_m, "Monthly"))

            all_events.append(row)

    if not all_events:
        st.warning("No historical RUN events found")
        st.stop()

    events_df = pd.DataFrame(all_events)

    # =================================================
    # STOCK PERSONALITY SUMMARY
    # =================================================
    summary = events_df.groupby("Stock").agg({
        "Daily_RSI": "mean",
        "Daily_ADX": "mean",
        "Price>EMA20": "mean",
        "Weekly_Trend": lambda x: (x == "Up").mean(),
        "Monthly_Trend": lambda x: (x == "Up").mean()
    }).reset_index()

    summary.rename(columns={
        "Daily_RSI": "Avg_RSI",
        "Daily_ADX": "Avg_ADX",
        "Price>EMA20": "EMA20_Above_%"
    }, inplace=True)

    # =================================================
    # UI TABS
    # =================================================
    tab1, tab2 = st.tabs(["📊 Stock Personality", "🧾 Raw Events"])

    with tab1:
        st.subheader("📊 Learned Stock Behavior")
        st.dataframe(summary, use_container_width=True)

    with tab2:
        st.subheader("🧾 Historical RUN Events")
        st.dataframe(events_df, use_container_width=True)
