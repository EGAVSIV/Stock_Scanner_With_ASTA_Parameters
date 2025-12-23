import os
import pandas as pd
import numpy as np
import streamlit as st
import talib

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(
    page_title="Back Testing Multi-Timeframe RUN Stock Analyzer",
    layout="wide",
    page_icon="🚀"
)

st.title("🚀Back Testing Multi-Timeframe Stock RUN Analyzer")
st.caption("Hourly Momentum + Daily & Weekly Confirmation")

# =====================================================
# DATA PATHS
# =====================================================
DATA_1H = "stock_data_1H"
DATA_D  = "stock_data_D"
DATA_W  = "stock_data_W"

# =====================================================
# BULLETPROOF DATA LOADER (NO CACHE)
# =====================================================
def load_stock(folder, symbol):
    path = os.path.join(folder, f"{symbol}.parquet")
    if not os.path.exists(path):
        return None

    df = pd.read_parquet(path)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Detect datetime safely
    dt_col = None
    for c in ["datetime", "date", "timestamp"]:
        if c in df.columns:
            dt_col = c
            break

    if dt_col:
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
        df = df.sort_values(dt_col)
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
    else:
        # Absolute fallback (never crash)
        df["__order__"] = range(len(df))
        df = df.sort_values("__order__")

    # Validate OHLC
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            return None

    return df.reset_index(drop=True)

# =====================================================
# SAFE INDICATOR CHECK
# =====================================================
def safe_last(series):
    if series is None or len(series) == 0:
        return None
    val = series.iloc[-1]
    if pd.isna(val):
        return None
    return val

# =====================================================
# BEST MA SELECTOR
# =====================================================
def best_ma(df):
    close = df["close"]

    mas = {
        "EMA20": talib.EMA(close, 20),
        "EMA50": talib.EMA(close, 50),
        "SMA20": talib.SMA(close, 20),
        "SMA50": talib.SMA(close, 50),
    }

    slopes = {}
    for k, v in mas.items():
        slopes[k] = v.diff().iloc[-1] if v is not None and not v.isna().all() else -999

    best = max(slopes, key=slopes.get)
    return best, mas[best]

# =====================================================
# HOURLY RUN DETECTION
# =====================================================
def hourly_run(df):
    if len(df) < 10:
        return False, 0

    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    pct_move = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100

    macd, _, macd_hist = talib.MACD(close)
    rsi = talib.RSI(close, 14)
    adx = talib.ADX(high, low, close, 14)

    macd_last = safe_last(macd_hist)
    rsi_last  = safe_last(rsi)
    adx_last  = safe_last(adx)

    if macd_last is None or rsi_last is None or adx_last is None:
        return False, 0

    return (
        pct_move >= 2 and
        macd_last > macd_hist.iloc[-2] and
        rsi_last > 55 and
        adx_last > adx.iloc[-2]
    ), round(pct_move, 2)

# =====================================================
# DAILY CONFIRMATION
# =====================================================
def daily_confirm(df):
    ma_name, ma = best_ma(df)
    rsi = talib.RSI(df["close"], 14)

    rsi_last = safe_last(rsi)
    if rsi_last is None:
        return False, ma_name

    return (
        df["close"].iloc[-1] > ma.iloc[-1] and
        rsi_last > 50
    ), ma_name

# =====================================================
# WEEKLY CONFIRMATION
# =====================================================
def weekly_confirm(df):
    ema20 = talib.EMA(df["close"], 20)
    ema50 = talib.EMA(df["close"], 50)
    adx   = talib.ADX(df["high"], df["low"], df["close"], 14)

    if safe_last(adx) is None:
        return False

    return ema20.iloc[-1] > ema50.iloc[-1] and adx.iloc[-1] > 20

# =====================================================
# STOCK SELECTION
# =====================================================
symbols = sorted([
    f.replace(".parquet", "")
    for f in os.listdir(DATA_1H)
    if f.endswith(".parquet")
])

selected = st.multiselect(
    "📌 Select Stocks",
    symbols,
    default=symbols[:10]
)

# =====================================================
# SCAN
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
        st.warning("No RUN stocks found")
