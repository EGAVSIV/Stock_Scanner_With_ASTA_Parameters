import os
import pandas as pd
import streamlit as st
import talib
import plotly.express as px

# ==============================
# STREAMLIT CONFIG
# ==============================
st.set_page_config(
    page_title="Multi-Timeframe Stock Screener",
    layout="wide"
)

st.title("📊 Multi-Timeframe Stock Screener Dashboard")

# ==============================
# TIMEFRAME → FOLDER MAP
# ==============================
TIMEFRAMES = {
    "15 Min": "stock_data_15",
    "1 Hour": "stock_data_1H",
    "Daily": "stock_data_D",
    "Weekly": "stock_data_W",
    "Monthly": "stock_data_M",
}

# ==============================
# DATA LOADER
# ==============================
@st.cache_data(show_spinner=False)
def load_parquet_data(folder):
    data = {}
    if not os.path.exists(folder):
        return data

    for f in os.listdir(folder):
        if not f.endswith(".parquet"):
            continue

        symbol = f.replace(".parquet", "")
        df = pd.read_parquet(os.path.join(folder, f))

        # Normalize datetime index
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").set_index("datetime")

        # Ensure OHLC
        needed = {"open", "high", "low", "close"}
        if not needed.issubset(df.columns):
            continue

        data[symbol] = df

    return data

# ==============================
# SCREENER 1: RSI MARKET PULSE
# ==============================
def rsi_market_pulse(df, period=14):
    if len(df) < period:
        return None

    rsi = talib.RSI(df["close"], period)
    val = rsi.iloc[-1]

    if pd.isna(val):
        return None

    if val > 60:
        zone = "RSI > 60"
    elif val < 40:
        zone = "RSI < 40"
    else:
        zone = "RSI 40–60"

    return round(val, 2), zone

# ==============================
# SCREENER 2: VOLUME SHOCKER
# ==============================
def volume_shocker(df):
    if len(df) < 20:
        return False

    vol_sma10 = df["volume"].rolling(10).mean()
    last, prev = df.iloc[-1], df.iloc[-2]

    if last["volume"] > 2 * vol_sma10.iloc[-1]:
        if prev["close"] * 0.95 <= last["close"] <= prev["close"] * 1.05:
            return True
    return False

# ==============================
# SCREENER 3: NRB-7
# ==============================
def nrb_7(df):
    if len(df) < 20:
        return False

    ref = df.iloc[-8]
    hi = df["high"].iloc[-7:-1].max()
    lo = df["low"].iloc[-7:-1].min()

    return ref["high"] > hi and ref["low"] < lo

# ==============================
# SCREENER 4: CANDLESTICK PATTERNS
# ==============================
CANDLE_PATTERNS = {
    "Doji": talib.CDLDOJI,
    "Hammer": talib.CDLHAMMER,
    "Shooting Star": talib.CDLSHOOTINGSTAR,
    "Engulfing": talib.CDLENGULFING,
    "Morning Star": talib.CDLMORNINGSTAR,
    "Evening Star": talib.CDLEVENINGSTAR,
    "Marubozu": talib.CDLMARUBOZU,
}

def candlestick_patterns(df):
    found = []
    for name, func in CANDLE_PATTERNS.items():
        val = func(df["open"], df["high"], df["low"], df["close"]).iloc[-1]
        if val != 0:
            found.append(f"{name} ({'Bullish' if val > 0 else 'Bearish'})")
    return found

# ==============================
# SIDEBAR CONTROLS
# ==============================
tf = st.sidebar.selectbox("Select Timeframe", list(TIMEFRAMES.keys()))
screener = st.sidebar.selectbox(
    "Select Screener",
    [
        "RSI Market Pulse",
        "Volume Shocker",
        "NRB-7 Breakout",
        "Candlestick Patterns",
    ]
)

run = st.sidebar.button("▶ Run Scan")

# ==============================
# MAIN EXECUTION
# ==============================
if run:
    folder = TIMEFRAMES[tf]
    data = load_parquet_data(folder)

    if not data:
        st.warning("No data found for selected timeframe.")
        st.stop()

    results = []

    for symbol, df in data.items():

        if screener == "RSI Market Pulse":
            r = rsi_market_pulse(df)
            if r:
                results.append({
                    "Symbol": symbol,
                    "RSI": r[0],
                    "Zone": r[1]
                })

        elif screener == "Volume Shocker":
            if volume_shocker(df):
                results.append({"Symbol": symbol})

        elif screener == "NRB-7 Breakout":
            if nrb_7(df):
                results.append({"Symbol": symbol})

        elif screener == "Candlestick Patterns":
            patterns = candlestick_patterns(df)
            for p in patterns:
                results.append({
                    "Symbol": symbol,
                    "Pattern": p
                })

    if not results:
        st.info("No stocks matched.")
        st.stop()

    df_res = pd.DataFrame(results)
    st.success(f"Stocks Found: {len(df_res)}")

    st.dataframe(df_res, use_container_width=True)

    # Optional summary chart
    if screener == "RSI Market Pulse":
        fig = px.histogram(df_res, x="Zone", title="RSI Zone Distribution")
        st.plotly_chart(fig, use_container_width=True)
