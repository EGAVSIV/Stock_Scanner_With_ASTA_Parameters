import os
import pandas as pd
import numpy as np
import streamlit as st
import talib

# =====================================================
# CONFIG
# =====================================================
st.set_page_config("Behavioral Stock Intelligence", layout="wide", page_icon="🧠")
st.title("🧠 Multi-Timeframe Behavioral Stock Intelligence")

DATA_D = "stock_data_D"
DATA_W = "stock_data_W"
DATA_M = "stock_data_M"

# =====================================================
# DATA LOADER
# =====================================================
def load_stock(folder, sym):
    p = os.path.join(folder, f"{sym}.parquet")
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
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
# DAILY EVENT DETECTION
# =====================================================
def detect_events(df):
    move = (df["close"].shift(-3) - df["close"]) / df["close"] * 100
    return df[(move >= 4)].index

# =====================================================
# DAILY SNAPSHOT (START OF MOVE)
# =====================================================
def daily_snapshot(df, i):
    c = df["close"]
    h, l = df["high"], df["low"]

    rsi = talib.RSI(c, 14)
    adx = talib.ADX(h, l, c, 14)
    pdi = talib.PLUS_DI(h, l, c, 14)
    mdi = talib.MINUS_DI(h, l, c, 14)

    macd, sig, hist = talib.MACD(c)
    ema20 = talib.EMA(c, 20)
    sma20 = talib.SMA(c, 20)
    bu, bm, bl = talib.BBANDS(c, 20)

    if any(pd.isna(x.iloc[i]) for x in [rsi, adx, macd, ema20]):
        return None

    return {
        "RSI": rsi.iloc[i],
        "ADX": adx.iloc[i],
        "DI_Positive": pdi.iloc[i] > mdi.iloc[i],
        "MACD_Up": macd.iloc[i] > sig.iloc[i] and hist.iloc[i] > hist.iloc[i-1],
        "PriceAboveMA": c.iloc[i] > max(ema20.iloc[i], sma20.iloc[i]),
        "BB_Expansion": (bu.iloc[i] - bl.iloc[i]) > (bu.iloc[i-3] - bl.iloc[i-3])
    }

# =====================================================
# WEEKLY / MONTHLY SNAPSHOT (LATEST)
# =====================================================
def higher_tf_snapshot(df):
    c = df["close"]
    rsi = talib.RSI(c, 14)
    macd, sig, _ = talib.MACD(c)
    ema20 = talib.EMA(c, 20)
    ema50 = talib.EMA(c, 50)

    if pd.isna(rsi.iloc[-1]) or pd.isna(macd.iloc[-1]):
        return None

    return {
        "RSI_Above_50": rsi.iloc[-1] > 50,
        "MACD_Above_Zero": macd.iloc[-1] > 0,
        "MACD_Bullish": macd.iloc[-1] > sig.iloc[-1],
        "PriceAboveEMA": c.iloc[-1] > ema20.iloc[-1] and c.iloc[-1] > ema50.iloc[-1]
    }

# =====================================================
# VERDICT BUILDER (HISTORICAL LEARNING)
# =====================================================
def build_verdict(events_df):
    v = {}
    v["Daily_RSI>60"] = (events_df["RSI"] > 60).mean()
    v["Daily_ADX>20"] = (events_df["ADX"] > 20).mean()
    v["Daily_DI+"] = events_df["DI_Positive"].mean()
    v["Daily_MACD_Up"] = events_df["MACD_Up"].mean()
    v["Daily_Price>MA"] = events_df["PriceAboveMA"].mean()
    v["Daily_BB_Expand"] = events_df["BB_Expansion"].mean()
    return v

# =====================================================
# CONFIDENCE ENGINE
# =====================================================
def confidence(verdict, daily_now, weekly_now, monthly_now):
    score = 0

    # DAILY (40)
    daily_rules = [
        daily_now["RSI"] > 60,
        daily_now["ADX"] > 20,
        daily_now["DI_Positive"],
        daily_now["MACD_Up"],
        daily_now["PriceAboveMA"]
    ]
    score += 40 * (sum(daily_rules) / len(daily_rules))

    # WEEKLY (35)
    weekly_rules = list(weekly_now.values())
    score += 35 * (sum(weekly_rules) / len(weekly_rules))

    # MONTHLY (25)
    monthly_rules = list(monthly_now.values())
    score += 25 * (sum(monthly_rules) / len(monthly_rules))

    return round(score, 1)

# =====================================================
# UI
# =====================================================
# =====================================================
# STOCK SELECTION (WITH SELECT ALL)
# =====================================================
symbols = sorted([f.replace(".parquet", "") for f in os.listdir(DATA_D)])

col1, col2 = st.columns([1, 5])

with col1:
    select_all = st.checkbox("✅ Select All")

with col2:
    if select_all:
        selected_symbols = st.multiselect(
            "📌 Selected Stocks",
            symbols,
            default=symbols
        )
    else:
        selected_symbols = st.multiselect(
            "📌 Selected Stocks",
            symbols,
            default=symbols[:5]
        )


if st.button("Run Behavioral Backtest"):
    output = []

    for s in selected_symbols:

        d = load_stock(DATA_D, s)
        w = load_stock(DATA_W, s)
        m = load_stock(DATA_M, s)
        if any(x is None for x in [d, w, m]):
            continue

        idx = detect_events(d)
        rows = []
        for i in idx:
            snap = daily_snapshot(d, i)
            if snap:
                rows.append(snap)

        if len(rows) < 3:
            continue

        hist = pd.DataFrame(rows)
        verdict = build_verdict(hist)

        daily_now = daily_snapshot(d, len(d)-1)
        weekly_now = higher_tf_snapshot(w)
        monthly_now = higher_tf_snapshot(m)

        if None in [daily_now, weekly_now, monthly_now]:
            continue

        score = confidence(verdict, daily_now, weekly_now, monthly_now)

        output.append({
            "Stock": s,
            "Confidence %": score,
            "Daily Align %": round(np.mean(list(verdict.values()))*100,1),
            "Weekly Align": weekly_now,
            "Monthly Align": monthly_now
        })

    if output:
        st.dataframe(pd.DataFrame(output), use_container_width=True)
    else:
        st.warning("No strong behavioral alignment found")
