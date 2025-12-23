import os
import pandas as pd
import numpy as np
import streamlit as st
import talib

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(
    page_title="Multi-Timeframe RUN Stock Analyzer",
    layout="wide",
    page_icon="🚀"
)

st.title("🚀 Multi-Timeframe Stock RUN Analyzer")

# =====================================================
# DATA PATHS
# =====================================================
DATA_1H = "stock_data_1H"
DATA_D  = "stock_data_D"
DATA_W  = "stock_data_W"

# =====================================================
# SAFE & BULLETPROOF DATA LOADER
# =====================================================
@st.cache_data(show_spinner=False)
def load_stock(folder, symbol):
    path = os.path.join(folder, f"{symbol}.parquet")
    if not os.path.exists(path):
        return None

    df = pd.read_parquet(path)

    # -------- Normalize column names --------
    df.columns = [c.strip().lower() for c in df.columns]

    # -------- Detect datetime --------
    dt_candidates = ["datetime", "date", "timestamp"]
    dt_col = next((c for c in dt_candidates if c in df.columns), None)

    if dt_col:
        df[dt_col] = pd.to_datetime(df[dt_col])
        df = df.sort_values(dt_col)
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
        df = df.reset_index(drop=True)
    else:
        # No datetime → create synthetic index (safe fallback)
        df["synthetic_dt"] = pd.RangeIndex(len(df))
        df = df.sort_values("synthetic_dt")

    # -------- Validate OHLC --------
    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(df.columns):
        return None

    return df.reset_index(drop=True)

# =====================================================
# BEST MA SELECTOR
# =====================================================
def best_ma(df):
    mas = {
        "EMA20": talib.EMA(df["close"], 20),
        "EMA50": talib.EMA(df["close"], 50),
        "SMA20": talib.SMA(df["close"], 20),
        "SMA50": talib.SMA(df["close"], 50),
    }

    slopes = {k: v.diff().iloc[-1] if not v.isna().all() else -999 for k, v in mas.items()}
    best = max(slopes, key=slopes.get)

    return best, mas[best]

# =====================================================
# HOURLY RUN DETECTION
# =====================================================
def hourly_run(df):
    if len(df) < 6:
        return False, 0

    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    pct_move = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100

    macd, macd_signal, macd_hist = talib.MACD(close)
    rsi = talib.RSI(close, 14)
    adx = talib.ADX(high, low, close, 14)

    if macd_hist.isna().iloc[-1] or adx.isna().iloc[-1]:
        return False, 0

    return (
        pct_move >= 4 and
        macd_hist.iloc[-1] > macd_hist.iloc[-2] and
        rsi.iloc[-1] > 55 and
        adx.iloc[-1] > adx.iloc[-2]
    ), round(pct_move, 2)

# =====================================================
# DAILY CONFIRMATION
# =====================================================
def daily_confirm(df):
    ma_name, ma = best_ma(df)
    rsi = talib.RSI(df["close"], 14)

    return (
        df["close"].iloc[-1] > ma.iloc[-1] and
        rsi.iloc[-1] > 50
    ), ma_name

# =====================================================
# WEEKLY CONFIRMATION
# =====================================================
def weekly_confirm(df):
    ema20 = talib.EMA(df["close"], 20)
    ema50 = talib.EMA(df["close"], 50)
    adx = talib.ADX(df["high"], df["low"], df["close"], 14)

    return (
        ema20.iloc[-1] > ema50.iloc[-1] and
        adx.iloc[-1] > 20
    )

# =====================================================
# STOCK LIST
# =====================================================
symbols = sorted([
    f.replace(".parquet", "")
    for f in os.listdir(DATA_1H)
    if f.endswith(".parquet")
])

selected = st.multiselect("📌 Select Stocks", symbols, default=symbols[:10])

# =====================================================
# RUN SCAN
# =====================================================
if st.button("🔍 Scan RUN Stocks"):
    results = []

    for sym in selected:
        df1h = load_stock(DATA_1H, sym)
        dfd  = load_stock(DATA_D, sym)
        dfw  = load_stock(DATA_W, sym)

        if df1h is None or dfd is None or dfw is None:
            continue

        run_ok, move = hourly_run(df1h)
        d_ok, ma_used = daily_confirm(dfd)
        w_ok = weekly_confirm(dfw)

        if run_ok and d_ok and w_ok:
            results.append({
                "Stock": sym,
                "Hourly Move %": move,
                "Daily MA": ma_used,
                "Daily Trend": "✔",
                "Weekly Trend": "✔",
                "Reason": "Hourly momentum + HTF alignment"
            })

    if results:
        st.success(f"🔥 {len(results)} RUN stocks found")
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    else:
        st.warning("No RUN stocks detected")
