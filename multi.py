import os
import pandas as pd
import streamlit as st
import talib
import plotly.express as px

# ==================================================
# STREAMLIT CONFIG
# ==================================================
st.set_page_config("Multi-Timeframe Stock Screener", layout="wide")
st.title("📊 Multi-Timeframe Stock Screener")

# ==================================================
# TIMEFRAME → DATA FOLDER
# ==================================================
TIMEFRAMES = {
    "15 Min": "stock_data_15",
    "1 Hour": "stock_data_1H",
    "Daily": "stock_data_D",
    "Weekly": "stock_data_W",
    "Monthly": "stock_data_M",
}

# ==================================================
# DATA LOADER
# ==================================================
@st.cache_data(show_spinner=False)
def load_data(folder):
    data = {}
    if not os.path.exists(folder):
        return data

    for f in os.listdir(folder):
        if not f.endswith(".parquet"):
            continue

        sym = f.replace(".parquet", "")
        df = pd.read_parquet(os.path.join(folder, f))

        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").set_index("datetime")

        needed = {"open", "high", "low", "close", "volume"}
        if not needed.issubset(df.columns):
            continue

        data[sym] = df

    return data

# ==================================================
# =============== SCANNERS =========================
# ==================================================

def rsi_market_pulse(df):
    if len(df) < 14: return None
    rsi = talib.RSI(df["close"], 14).iloc[-1]
    if rsi > 60: zone = "RSI > 60"
    elif rsi < 40: zone = "RSI < 40"
    else: zone = "RSI 40–60"
    return round(rsi, 2), zone

def volume_shocker(df):
    if len(df) < 20: return False
    vol_sma = df["volume"].rolling(10).mean()
    last, prev = df.iloc[-1], df.iloc[-2]
    return last["volume"] > 2 * vol_sma.iloc[-1] and prev["close"] * 0.95 <= last["close"] <= prev["close"] * 1.05

def nrb_7(df):
    if len(df) < 20: return False
    ref = df.iloc[-8]
    return ref["high"] > df["high"].iloc[-7:-1].max() and ref["low"] < df["low"].iloc[-7:-1].min()

def counter_attack(df):
    if len(df) < 2: return None
    prev, curr = df.iloc[-2], df.iloc[-1]
    mid = (prev["open"] + prev["close"]) / 2
    if prev["close"] < prev["open"] and curr["close"] > curr["open"]:
        if curr["open"] < prev["close"] * 0.97 and curr["close"] >= mid:
            return "Bullish"
    if prev["close"] > prev["open"] and curr["close"] < curr["open"]:
        if curr["open"] > prev["close"] * 1.03 and curr["close"] <= mid:
            return "Bearish"
    return None

def breakaway_gap(df):
    if len(df) < 50: return None
    ema20 = talib.EMA(df["close"], 20)
    ema50 = talib.EMA(df["close"], 50)
    prev, curr = df.iloc[-2], df.iloc[-1]
    if curr["open"] > prev["high"] * 1.005 and curr["low"] > prev["high"] and ema20.iloc[-1] < ema50.iloc[-1]:
        return "Bullish Breakaway Gap"
    if curr["open"] < prev["low"] * 0.995 and curr["high"] < prev["low"] and ema20.iloc[-1] > ema50.iloc[-1]:
        return "Bearish Breakaway Gap"
    return None

def macd_hook_up(df):
    macd, _, hist = talib.MACD(df["close"], 12, 26, 9)
    return "MACD Hook Up" if macd.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2] else None

def macd_hook_down(df):
    macd, _, hist = talib.MACD(df["close"], 12, 26, 9)
    return "MACD Hook Down" if macd.iloc[-1] < 0 and hist.iloc[-1] < hist.iloc[-2] else None

def macd_histogram_divergence(df):
    _, _, hist = talib.MACD(df["close"], 12, 26, 9)
    if df["low"].iloc[-1] < df["low"].iloc[-5] and hist.iloc[-1] > hist.iloc[-5]:
        return "Bullish Histogram Divergence"
    if df["high"].iloc[-1] > df["high"].iloc[-5] and hist.iloc[-1] < hist.iloc[-5]:
        return "Bearish Histogram Divergence"
    return None

def ema50_stoch_oversold(df):
    ema50 = talib.EMA(df["close"], 50)
    k, d = talib.STOCH(df["high"], df["low"], df["close"])
    near = abs(df["close"].iloc[-1] - ema50.iloc[-1]) / ema50.iloc[-1] <= 0.005
    cross = k.iloc[-2] < d.iloc[-2] and k.iloc[-1] > d.iloc[-1] and k.iloc[-1] < 20
    return "EMA50 + Stoch Oversold Buy" if near and cross else None

def dark_cloud_cover(df):
    prev, curr = df.iloc[-2], df.iloc[-1]
    if prev["close"] > prev["open"] and curr["open"] > prev["close"] * 1.01 and curr["close"] < prev["low"]:
        return "Dark Cloud Cover (Bearish)"
    return None

def morning_star_bottom(df):
    ema50 = talib.EMA(df["close"], 50)
    if df["close"].iloc[-1] < ema50.iloc[-1] and talib.CDLMORNINGSTAR(df["open"], df["high"], df["low"], df["close"]).iloc[-1] > 0:
        return "Morning Star (Bottom)"
    return None

def evening_star_top(df):
    ema50 = talib.EMA(df["close"], 50)
    if df["close"].iloc[-1] > ema50.iloc[-1] and talib.CDLEVENINGSTAR(df["open"], df["high"], df["low"], df["close"]).iloc[-1] < 0:
        return "Evening Star (Top)"
    return None

def bullish_gsas(df_tf, df_htf):
    rsi = talib.RSI(df_tf["close"], 14)
    adx = talib.ADX(df_tf["high"], df_tf["low"], df_tf["close"], 14)
    ubb, _, _ = talib.BBANDS(df_tf["close"], 20)
    macd, sig, _ = talib.MACD(df_htf["close"], 12, 26, 9)
    ema20 = talib.EMA(df_htf["close"], 20)
    return "Bullish GSAS" if rsi.iloc[-1] > 60 and ubb.iloc[-1] > ubb.iloc[-2] and adx.iloc[-1] > adx.iloc[-2] and macd.iloc[-1] > sig.iloc[-1] and df_htf["close"].iloc[-1] > ema20.iloc[-1] else None

def bearish_gsas(df_tf, df_htf):
    rsi = talib.RSI(df_tf["close"], 14)
    adx = talib.ADX(df_tf["high"], df_tf["low"], df_tf["close"], 14)
    _, _, lbb = talib.BBANDS(df_tf["close"], 20)
    macd, sig, _ = talib.MACD(df_htf["close"], 12, 26, 9)
    ema20 = talib.EMA(df_htf["close"], 20)
    return "Bearish GSAS" if rsi.iloc[-1] < 60 and lbb.iloc[-1] < lbb.iloc[-2] and adx.iloc[-1] > adx.iloc[-2] and macd.iloc[-1] < sig.iloc[-1] and df_htf["close"].iloc[-1] < ema20.iloc[-1] else None

# ==================================================
# SIDEBAR
# ==================================================
tf = st.sidebar.selectbox("Timeframe", list(TIMEFRAMES.keys()))

scanner = st.sidebar.selectbox(
    "Scanner",
    [
        "RSI Market Pulse", "Volume Shocker", "NRB-7 Breakout",
        "Counter Attack", "Breakaway Gaps",
        "MACD Hook Up", "MACD Hook Down",
        "MACD Histogram Divergence",
        "EMA50 + Stoch Oversold",
        "Dark Cloud Cover",
        "Morning Star (Bottom)", "Evening Star (Top)",
        "Bullish GSAS", "Bearish GSAS",
    ]
)

run = st.sidebar.button("▶ Run Scan")

# ==================================================
# MAIN EXECUTION
# ==================================================
if run:
    data = load_data(TIMEFRAMES[tf])
    if not data:
        st.warning("No data found")
        st.stop()

    results = []

    data_htf = None
    if scanner in ["Bullish GSAS", "Bearish GSAS"]:
        htf_map = {"15 Min": "1 Hour", "1 Hour": "Daily", "Daily": "Weekly", "Weekly": "Monthly"}
        data_htf = load_data(TIMEFRAMES[htf_map[tf]])

    for sym, df in data.items():

        sig = None

        if scanner == "RSI Market Pulse":
            r = rsi_market_pulse(df)
            if r: results.append({"Symbol": sym, "RSI": r[0], "Zone": r[1]})

        elif scanner == "Bullish GSAS" and data_htf and sym in data_htf:
            sig = bullish_gsas(df, data_htf[sym])

        elif scanner == "Bearish GSAS" and data_htf and sym in data_htf:
            sig = bearish_gsas(df, data_htf[sym])

        elif scanner == "MACD Hook Up":
            sig = macd_hook_up(df)

        elif scanner == "MACD Hook Down":
            sig = macd_hook_down(df)

        elif scanner == "MACD Histogram Divergence":
            sig = macd_histogram_divergence(df)

        elif scanner == "EMA50 + Stoch Oversold":
            sig = ema50_stoch_oversold(df)

        elif scanner == "Dark Cloud Cover":
            sig = dark_cloud_cover(df)

        elif scanner == "Morning Star (Bottom)":
            sig = morning_star_bottom(df)

        elif scanner == "Evening Star (Top)":
            sig = evening_star_top(df)

        if sig:
            results.append({"Symbol": sym, "Signal": sig})

    if not results:
        st.info("No stocks matched")
        st.stop()

    df_res = pd.DataFrame(results)
    st.success(f"Stocks Found: {len(df_res)}")
    st.dataframe(df_res, use_container_width=True)

    if scanner == "RSI Market Pulse":
        fig = px.histogram(df_res, x="Zone", title="RSI Zone Distribution")
        st.plotly_chart(fig, use_container_width=True)
