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
st.caption("Hourly Momentum + Daily & Weekly Confirmation")

# =====================================================
# DATA PATHS
# =====================================================
DATA_1H = "stock_data_1H"
DATA_D  = "stock_data_D"
DATA_W  = "stock_data_W"

# =====================================================
# SAFE DATA LOADER
# =====================================================
@st.cache_data(show_spinner=False)
def load_stock(folder, symbol):
    path = os.path.join(folder, f"{symbol}.parquet")
    if not os.path.exists(path):
        return None

    df = pd.read_parquet(path)

    # -------- Detect datetime column or index ----------
    possible_cols = ["Datetime", "datetime", "Date", "date", "Timestamp", "timestamp"]
    dt_col = None

    for c in possible_cols:
        if c in df.columns:
            dt_col = c
            break

    if dt_col is None and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        dt_col = df.columns[0]

    if dt_col is None:
        raise ValueError(f"No Datetime column found in {symbol}")

    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col).reset_index(drop=True)

    return df

# =====================================================
# BEST MA SELECTOR (STOCK-SPECIFIC)
# =====================================================
def select_best_ma(df):
    ma_map = {
        "EMA20": talib.EMA(df["Close"], 20),
        "EMA50": talib.EMA(df["Close"], 50),
        "SMA20": talib.SMA(df["Close"], 20),
        "SMA50": talib.SMA(df["Close"], 50),
    }

    slopes = {}
    for k, v in ma_map.items():
        if v.isna().all():
            slopes[k] = -999
        else:
            slopes[k] = v.diff().iloc[-1]

    best = max(slopes, key=slopes.get)
    return best, ma_map[best]

# =====================================================
# HOURLY RUN DETECTION (TRIGGER)
# =====================================================
def detect_hourly_run(df):
    if len(df) < 10:
        return False, None

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    pct_move = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100

    macd, macd_signal, macd_hist = talib.MACD(close)
    rsi = talib.RSI(close, 14)
    adx = talib.ADX(high, low, close, 14)

    bb_upper, bb_mid, bb_lower = talib.BBANDS(close, 20)
    bb_expanding = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) > \
                   (bb_upper.iloc[-5] - bb_lower.iloc[-5])

    condition = (
        pct_move >= 4 and
        macd_hist.iloc[-1] > macd_hist.iloc[-2] and
        rsi.iloc[-1] > 55 and
        adx.iloc[-1] > adx.iloc[-2] and
        bb_expanding
    )

    return condition, round(pct_move, 2)

# =====================================================
# DAILY CONFIRMATION
# =====================================================
def daily_confirmation(df):
    best_ma_name, best_ma = select_best_ma(df)
    rsi = talib.RSI(df["Close"], 14)

    confirmed = (
        df["Close"].iloc[-1] > best_ma.iloc[-1] and
        rsi.iloc[-1] > 50
    )

    return confirmed, best_ma_name

# =====================================================
# WEEKLY CONFIRMATION
# =====================================================
def weekly_confirmation(df):
    ema20 = talib.EMA(df["Close"], 20)
    ema50 = talib.EMA(df["Close"], 50)
    adx   = talib.ADX(df["High"], df["Low"], df["Close"], 14)

    return (
        ema20.iloc[-1] > ema50.iloc[-1] and
        adx.iloc[-1] > 20
    )

# =====================================================
# STOCK SELECTION
# =====================================================
symbols = sorted([
    f.replace(".parquet", "")
    for f in os.listdir(DATA_1H)
    if f.endswith(".parquet")
])

selected_stocks = st.multiselect(
    "📌 Select Stocks",
    symbols,
    default=symbols[:15]
)

# =====================================================
# SCAN BUTTON
# =====================================================
if st.button("🔍 Scan RUN Stocks"):
    results = []

    for sym in selected_stocks:
        df1h = load_stock(DATA_1H, sym)
        dfd  = load_stock(DATA_D, sym)
        dfw  = load_stock(DATA_W, sym)

        if df1h is None or dfd is None or dfw is None:
            continue

        run_ok, move = detect_hourly_run(df1h)
        daily_ok, ma_used = daily_confirmation(dfd)
        weekly_ok = weekly_confirmation(dfw)

        if run_ok and daily_ok and weekly_ok:
            results.append({
                "Stock": sym,
                "Hourly Move %": move,
                "Daily MA Used": ma_used,
                "Hourly Trigger": "✔ Momentum Burst",
                "Daily Trend": "✔ Confirmed",
                "Weekly Trend": "✔ Confirmed",
                "Reason": "4–5% move in ≤5 candles with HTF alignment"
            })

    if results:
        df_out = pd.DataFrame(results)
        st.success(f"🔥 {len(df_out)} RUN candidates found")
        st.dataframe(df_out, use_container_width=True)
    else:
        st.warning("No RUN stocks found with current conditions")
