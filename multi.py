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
        if f.endswith(".parquet"):
            sym = f.replace(".parquet", "")
            df = pd.read_parquet(os.path.join(folder, f))

            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()

            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.sort_values("datetime").set_index("datetime")

            needed = {"open","high","low","close","volume"}
            if not needed.issubset(df.columns):
                continue

            data[sym] = df

    return data

# ==================================================
# ======== SCANNERS (PURE FUNCTIONS) ================
# ==================================================

# --- RSI Market Pulse
def rsi_market_pulse(df):
    if len(df) < 14:
        return None
    rsi = talib.RSI(df["close"], 14).iloc[-1]
    if rsi > 60: zone = "RSI > 60"
    elif rsi < 40: zone = "RSI < 40"
    else: zone = "RSI 40–60"
    return round(rsi,2), zone

# --- Volume Shocker
def volume_shocker(df):
    if len(df) < 20:
        return False
    vol_sma = df["volume"].rolling(10).mean()
    last, prev = df.iloc[-1], df.iloc[-2]
    return last["volume"] > 2 * vol_sma.iloc[-1] and prev["close"]*0.95 <= last["close"] <= prev["close"]*1.05

# --- NRB-7
def nrb_7(df):
    if len(df) < 20:
        return False
    ref = df.iloc[-8]
    return ref["high"] > df["high"].iloc[-7:-1].max() and ref["low"] < df["low"].iloc[-7:-1].min()

# --- Counter Attack
def counter_attack(df):
    if len(df) < 2:
        return None
    prev, curr = df.iloc[-2], df.iloc[-1]
    mid = (prev["open"] + prev["close"]) / 2

    if prev["close"] < prev["open"] and curr["close"] > curr["open"]:
        if curr["open"] < prev["close"] * 0.97 and curr["close"] >= mid:
            return "Bullish"
    if prev["close"] > prev["open"] and curr["close"] < curr["open"]:
        if curr["open"] > prev["close"] * 1.03 and curr["close"] <= mid:
            return "Bearish"
    return None

# --- Breakaway Gaps
def breakaway_gap(df):
    if len(df) < 50:
        return None
    df["EMA20"] = talib.EMA(df["close"], 20)
    df["EMA50"] = talib.EMA(df["close"], 50)

    prev, curr = df.iloc[-2], df.iloc[-1]

    if curr["open"] > prev["high"] * 1.005 and curr["low"] > prev["high"]:
        if curr["EMA20"] < curr["EMA50"]:
            return "Bullish Breakaway Gap"

    if curr["open"] < prev["low"] * 0.995 and curr["high"] < prev["low"]:
        if curr["EMA20"] > curr["EMA50"]:
            return "Bearish Breakaway Gap"

    return None

# --- RSI + ADX
def rsi_adx(df):
    if len(df) < 20:
        return None
    rsi = talib.RSI(df["close"],14).iloc[-1]
    adx = talib.ADX(df["high"],df["low"],df["close"],14).iloc[-1]

    if adx > 50 and rsi < 20:
        return "Bullish Reversal"
    if adx > 50 and rsi > 80:
        return "Bearish Reversal"
    return None

# --- RSI WM 60–40 (Multi-TF)
def rsi_wm(df_tf, df_w, df_m):
    r_tf = talib.RSI(df_tf["close"],14).iloc[-1]
    r_w  = talib.RSI(df_w["close"],14).iloc[-1]
    r_m  = talib.RSI(df_m["close"],14).iloc[-1]

    if r_w > 60 and r_m > 60 and r_tf < 40:
        return "Bullish WM Reversal"
    if r_w < 40 and r_m < 40 and r_tf > 60:
        return "Bearish WM Reversal"
    return None


def macd_market_pulse(df):
    if len(df) < 30:
        return None

    macd, signal, _ = talib.MACD(df["close"], 12, 26, 9)

    m, s = macd.iloc[-1], signal.iloc[-1]
    pm = macd.iloc[-2]

    if m > 0 and m > s and m > pm:
        return "Strong Bullish"
    if m > 0 and m > s and m < pm:
        return "Bullish Cooling"
    if m > 0 and m < s and m > pm:
        return "Bullish Reversal Watch"
    if m > 0 and m < s and m < pm:
        return "Weak Bullish"

    if m < 0 and m > s and m > pm:
        return "Bearish Reversal Watch"
    if m < 0 and m > s and m < pm:
        return "Weak Bearish"
    if m < 0 and m < s and m > pm:
        return "Bearish Recovery Attempt"
    if m < 0 and m < s and m < pm:
        return "Strong Bearish"

    return None

def macd_normal_divergence(df, lookback=30):
    if len(df) < lookback:
        return None

    macd, _, _ = talib.MACD(df["close"], 12, 26, 9)

    price_low1 = df["low"].iloc[-lookback:-15].min()
    price_low2 = df["low"].iloc[-15:].min()

    macd_low1 = macd.iloc[-lookback:-15].min()
    macd_low2 = macd.iloc[-15:].min()

    if price_low2 < price_low1 and macd_low2 > macd_low1:
        return "Bullish ND"

    price_high1 = df["high"].iloc[-lookback:-15].max()
    price_high2 = df["high"].iloc[-15:].max()

    macd_high1 = macd.iloc[-lookback:-15].max()
    macd_high2 = macd.iloc[-15:].max()

    if price_high2 > price_high1 and macd_high2 < macd_high1:
        return "Bearish ND"

    return None


def macd_rd(df):
    if len(df) < 60:
        return None

    macd, _, _ = talib.MACD(df["close"], 12, 26, 9)
    latest = macd.iloc[-1]
    prev = macd.iloc[-2]
    max60 = macd.rolling(60).max().iloc[-1]

    if latest > prev and latest > 0 and max60 > 0:
        if (latest / max60) < 0.25:
            return "MACD RD (Compression)"

    return None


def third_wave_finder(df):
    if len(df) < 60:
        return False

    ema20 = talib.EMA(df["close"], 20)
    ema50 = talib.EMA(df["close"], 50)

    if ema20.iloc[-1] > ema50.iloc[-1] and ema20.iloc[-2] < ema50.iloc[-2]:
        low1 = df["low"].iloc[-30:].min()
        low2 = df["low"].iloc[-60:-30].min()
        if low1 > low2:
            return True

    return False

def macd_peak_bearish_divergence(df):
    if len(df) < 80:
        return None

    macd, _, _ = talib.MACD(df["close"], 12, 26, 9)

    price_high1 = df["high"].iloc[-60:-30].max()
    price_high2 = df["high"].iloc[-30:].max()

    macd_high1 = macd.iloc[-60:-30].max()
    macd_high2 = macd.iloc[-30:].max()

    if price_high2 > price_high1 and macd_high2 < macd_high1:
        return "Bearish MACD Peak Divergence"

    return None

def macd_base_bullish_divergence(df):
    if len(df) < 80:
        return None

    macd, _, _ = talib.MACD(df["close"], 12, 26, 9)

    price_low1 = df["low"].iloc[-60:-30].min()
    price_low2 = df["low"].iloc[-30:].min()

    macd_low1 = macd.iloc[-60:-30].min()
    macd_low2 = macd.iloc[-30:].min()

    if price_low2 < price_low1 and macd_low2 > macd_low1:
        return "Bullish MACD Base Divergence"

    return None
def trend_alignment(df):
    if len(df) < 100:
        return None

    ema20 = talib.EMA(df["close"], 20)
    ema50 = talib.EMA(df["close"], 50)
    ema100 = talib.EMA(df["close"], 100)

    if ema20.iloc[-1] > ema50.iloc[-1] > ema100.iloc[-1]:
        return "Strong Uptrend"

    if ema20.iloc[-1] < ema50.iloc[-1] < ema100.iloc[-1]:
        return "Strong Downtrend"

    return None

def pullback_to_ema(df):
    if len(df) < 60:
        return None

    ema20 = talib.EMA(df["close"], 20)
    ema50 = talib.EMA(df["close"], 50)

    last = df.iloc[-1]

    # Bullish pullback
    if ema20.iloc[-1] > ema50.iloc[-1]:
        if last["low"] <= ema20.iloc[-1] and last["close"] > ema20.iloc[-1]:
            return "Bullish EMA Pullback"

    # Bearish pullback
    if ema20.iloc[-1] < ema50.iloc[-1]:
        if last["high"] >= ema20.iloc[-1] and last["close"] < ema20.iloc[-1]:
            return "Bearish EMA Pullback"

    return None

def confluence_setup(df):
    if len(df) < 60:
        return None

    rsi = talib.RSI(df["close"], 14).iloc[-1]
    macd, sig, _ = talib.MACD(df["close"], 12, 26, 9)
    ema20 = talib.EMA(df["close"], 20)
    ema50 = talib.EMA(df["close"], 50)

    # Bullish confluence
    if (
        rsi > 50 and
        macd.iloc[-1] > sig.iloc[-1] and
        ema20.iloc[-1] > ema50.iloc[-1]
    ):
        return "Bullish Confluence"

    # Bearish confluence
    if (
        rsi < 50 and
        macd.iloc[-1] < sig.iloc[-1] and
        ema20.iloc[-1] < ema50.iloc[-1]
    ):
        return "Bearish Confluence"

    return None



# ==================================================
# SIDEBAR
# ==================================================
# ==================================================
# SIDEBAR
# ==================================================
tf = st.sidebar.selectbox(
    "Timeframe",
    list(TIMEFRAMES.keys())
)

scanner = st.sidebar.selectbox(
    "Scanner",
    [
        "RSI Market Pulse",
        "Volume Shocker",
        "NRB-7 Breakout",
        "Counter Attack",
        "Breakaway Gaps",
        "RSI + ADX",
        "RSI WM 60–40",
        "MACD Market Pulse",
        "MACD Normal Divergence",
        "MACD RD (4th Wave)",
        "Probable 3rd Wave",
        "MACD Bearish Peak Divergence",
        "MACD Bullish Base Divergence",
        "Trend Alignment (EMA)",
        "Pullback to EMA",
        "High Probability Confluence",

    ]
)

run = st.sidebar.button("▶ Run Scan")





# ==================================================
# MAIN EXECUTION
# ==================================================
if run:
    data = load_data(TIMEFRAMES[tf])
    if not data:
        st.warning("No data found.")
        st.stop()

    results = []

    # preload W/M if required
    if scanner == "RSI WM 60–40":
        data_w = load_data(TIMEFRAMES["Weekly"])
        data_m = load_data(TIMEFRAMES["Monthly"])

    for sym, df in data.items():

        if scanner == "RSI Market Pulse":
            r = rsi_market_pulse(df)
            if r:
                results.append({"Symbol": sym, "RSI": r[0], "Zone": r[1]})

        elif scanner == "Volume Shocker" and volume_shocker(df):
            results.append({"Symbol": sym})

        elif scanner == "NRB-7 Breakout" and nrb_7(df):
            results.append({"Symbol": sym})

        elif scanner == "Counter Attack":
            sig = counter_attack(df)
            if sig:
                results.append({"Symbol": sym, "Signal": sig})

        elif scanner == "MACD Market Pulse":
            sig = macd_market_pulse(df)
            if sig:
                results.append({"Symbol": sym, "State": sig})

        elif scanner == "MACD Normal Divergence":
            sig = macd_normal_divergence(df)
            if sig:
                results.append({"Symbol": sym, "Divergence": sig})

        elif scanner == "MACD RD (4th Wave)":
            sig = macd_rd(df)
            if sig:
                results.append({"Symbol": sym, "Signal": sig})

        elif scanner == "Probable 3rd Wave":
            if third_wave_finder(df):
                results.append({"Symbol": sym})

        elif scanner == "MACD Bearish Peak Divergence":
            sig = macd_peak_bearish_divergence(df)
            if sig:
                results.append({"Symbol": sym, "Signal": sig})

        elif scanner == "MACD Bullish Base Divergence":
            sig = macd_base_bullish_divergence(df)
            if sig:
                results.append({"Symbol": sym, "Signal": sig})

        elif scanner == "Trend Alignment (EMA)":
            sig = trend_alignment(df)
            if sig:
                results.append({"Symbol": sym, "Trend": sig}
        elif scanner == "Pullback to EMA":
            sig = pullback_to_ema(df)
            if sig:
                results.append({"Symbol": sym, "Setup": sig})

        elif scanner == "High Probability Confluence":
            sig = confluence_setup(df)
            if sig:
                results.append({"Symbol": sym, "Setup": sig})



        elif scanner == "Breakaway Gaps":
            sig = breakaway_gap(df)
            if sig:
                results.append({"Symbol": sym, "Signal": sig})

        elif scanner == "RSI + ADX":
            sig = rsi_adx(df)
            if sig:
                results.append({"Symbol": sym, "Signal": sig})

        elif scanner == "RSI WM 60–40":
            if sym in data_w and sym in data_m:
                sig = rsi_wm(df, data_w[sym], data_m[sym])
                if sig:
                    results.append({"Symbol": sym, "Signal": sig})

    if not results:
        st.info("No stocks matched.")
        st.stop()

    df_res = pd.DataFrame(results)
    st.success(f"Stocks Found: {len(df_res)}")
    st.dataframe(df_res, use_container_width=True)

    if scanner == "RSI Market Pulse":
        fig = px.histogram(df_res, x="Zone", title="RSI Zone Distribution")
        st.plotly_chart(fig, use_container_width=True)
