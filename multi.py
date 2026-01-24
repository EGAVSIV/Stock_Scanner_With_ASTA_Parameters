import os
import pandas as pd
import numpy as np
import sys
if sys.version_info >= (3, 13):
    import types
    imghdr = types.ModuleType("imghdr")
    imghdr.what = lambda *args, **kwargs: None
    sys.modules["imghdr"] = imghdr
import streamlit as st
import talib
import plotly.express as px
import hashlib


def hash_pwd(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()

USERS = st.secrets["users"]

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Login Required")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u in USERS and hash_pwd(p) == USERS[u]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()


def get_last_candle_by_tf(folder_path: str):
    last_dt = None

    if not os.path.isdir(folder_path):
        return None

    for f in os.listdir(folder_path):
        if not f.endswith(".parquet"):
            continue
        try:
            df = pd.read_parquet(os.path.join(folder_path, f))
            if df.empty:
                continue

            if isinstance(df.index, pd.DatetimeIndex):
                dt = df.index[-1]
            elif "datetime" in df.columns:
                dt = pd.to_datetime(df["datetime"]).iloc[-1]
            else:
                continue

            # Assume UTC → convert to IST
            if dt.tzinfo is None:
                dt = dt.tz_localize("UTC")
            else:
                dt = dt.tz_convert("UTC")

            dt = dt.tz_convert("Asia/Kolkata")

            if last_dt is None or dt > last_dt:
                last_dt = dt

        except Exception:
            continue

    return last_dt


# ==================================================
# STREAMLIT CONFIG
# ==================================================
st.set_page_config("Multi-Timeframe Stock Screener", layout="wide",page_icon="🧮")

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
# LAST CANDLE DATES (ALL TIMEFRAMES – IST)
# ==================================================
last_15m = get_last_candle_by_tf(TIMEFRAMES["15 Min"])
last_1h  = get_last_candle_by_tf(TIMEFRAMES["1 Hour"])
last_d   = get_last_candle_by_tf(TIMEFRAMES["Daily"])
last_w   = get_last_candle_by_tf(TIMEFRAMES["Weekly"])
last_m   = get_last_candle_by_tf(TIMEFRAMES["Monthly"])


st.title("📊 Multi-Timeframe Stock Screener")
# ==================================================
# TOP DATA REFRESH + LAST CANDLE INFO
# ==================================================
col1, col2 = st.columns([1, 6])

with col1:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.success("Fresh data loaded")
        st.rerun()

with col2:
    st.markdown(
        f"""
🕯 **Last Candle (IST)**  
⏱ **15 Min**: {last_15m.strftime('%d %b %Y %H:%M') if last_15m else 'NA'}  |  
⏰ **1 Hour**: {last_1h.strftime('%d %b %Y %H:%M') if last_1h else 'NA'}  |  
📅 **Daily**: {last_d.date() if last_d else 'NA'}  |  
📆 **Weekly**: {last_w.date() if last_w else 'NA'}  |  
🗓 **Monthly**: {last_m.date() if last_m else 'NA'}
""",
        unsafe_allow_html=False
    )

st.markdown("---")





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
    """
    NRB-7 + Breakout confirmation + Volume filter
    Returns:
        None → no signal
        String → Bullish / Bearish NRB breakout
    """

    if len(df) < 20:
        return None

    # NRB reference candle (7th from last)
    base = df.iloc[-7]

    # Next 6 candles (compression candles)
    inside = df.iloc[-6:-1]

    # Latest candle (breakout candle)
    last = df.iloc[-1]

    # --- NRB-7 condition ---
    is_nrb = (
        base["high"] > inside["high"].max() and
        base["low"]  < inside["low"].min()
    )

    if not is_nrb:
        return None

    # --- Volume filter ---
    avg_vol = df["volume"].rolling(10).mean().iloc[-2]
    if last["volume"] < 1.5 * avg_vol:
        return None

    # --- Breakout confirmation ---
    if last["close"] > base["high"]:
        return "NRB-7 Bullish Breakout + Volume"

    if last["close"] < base["low"]:
        return "NRB-7 Bearish Breakdown + Volume"

    return None



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
        return "Probabale Bearish Reversal"
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

def c_wave_finder(df):
    if len(df) < 60:
        return False

    ema20 = talib.EMA(df["close"], 20)
    ema50 = talib.EMA(df["close"], 50)
    rsi = talib.RSI(df["close"], 14)

    # fresh bearish crossover
    if ema20.iloc[-1] < ema50.iloc[-1] and ema20.iloc[-2] > ema50.iloc[-2]:
        high1 = df["high"].iloc[-30:].max()
        high2 = df["high"].iloc[-60:-30].max()

        # lower high
        if high1 < high2 and rsi.iloc[-1] > 40:
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

def macd_hook_up(df):
    if len(df) < 35:
        return None

    macd, signal, hist = talib.MACD(df["close"], 12, 26, 9)

    if (
        macd.iloc[-1] > 0 and
        macd.iloc[-2] < macd.iloc[-3] and
        macd.iloc[-1] > macd.iloc[-2] and
        hist.iloc[-1] > hist.iloc[-2]
    ):
        return "MACD Hook Up"

    return None

def macd_hook_down(df):
    if len(df) < 35:
        return None

    macd, signal, hist = talib.MACD(df["close"], 12, 26, 9)

    if (
        macd.iloc[-1] < 0 and
        macd.iloc[-2] > macd.iloc[-3] and
        macd.iloc[-1] < macd.iloc[-2] and
        hist.iloc[-1] < hist.iloc[-2]
    ):
        return "MACD Hook Down"

    return None


def macd_histogram_divergence(df):
    if len(df) < 50:
        return None

    _, _, hist = talib.MACD(df["close"], 12, 26, 9)

    price_low1 = df["low"].iloc[-40:-20].min()
    price_low2 = df["low"].iloc[-20:].min()

    hist_low1 = hist.iloc[-40:-20].min()
    hist_low2 = hist.iloc[-20:].min()

    if price_low2 < price_low1 and hist_low2 > hist_low1:
        return "Bullish Histogram Divergence"

    price_high1 = df["high"].iloc[-40:-20].max()
    price_high2 = df["high"].iloc[-20:].max()

    hist_high1 = hist.iloc[-40:-20].max()
    hist_high2 = hist.iloc[-20:].max()

    if price_high2 > price_high1 and hist_high2 < hist_high1:
        return "Bearish Histogram Divergence"

    return None

def ema50_stoch_oversold(df):
    if len(df) < 50:
        return None

    ema50 = talib.EMA(df["close"], 50)
    slowk, slowd = talib.STOCH(
        df["high"], df["low"], df["close"],
        fastk_period=14, slowk_period=3, slowd_period=3
    )

    price = df["close"].iloc[-1]
    ema_val = ema50.iloc[-1]

    near_ema = abs(price - ema_val) / ema_val <= 0.005

    stoch_cross = (
        slowk.iloc[-2] < slowd.iloc[-2] and
        slowk.iloc[-1] > slowd.iloc[-1] and
        slowk.iloc[-1] < 20
    )

    if near_ema and stoch_cross:
        return "EMA50 + Stoch Oversold Buy"

    return None

def dark_cloud_cover(df):
    if len(df) < 15:  # RSI needs at least 14 candles
        return None

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    # Previous candle must be bullish
    if prev["close"] <= prev["open"]:
        return None

    # Previous RSI condition
    rsi = talib.RSI(df["close"], 14)
    if rsi.iloc[-2] <= 60:
        return None

    # Gap up
    gap_up = curr["open"] > prev["close"]

    # Close below 50% of previous candle body
    mid = (prev["open"] + prev["close"]) / 2
    close_below_mid = curr["close"] < mid

    if gap_up and close_below_mid:
        return "Dark Cloud Cover (Bearish | RSI>60)"

    return None


def morning_star_bottom(df):
    if len(df) < 60:
        return None

    ema50 = talib.EMA(df["close"], 50)

    # downtrend condition
    if df["close"].iloc[-1] > ema50.iloc[-1]:
        return None

    pattern = talib.CDLMORNINGSTAR(
        df["open"], df["high"], df["low"], df["close"]
    ).iloc[-1]

    if pattern > 0:
        return "Morning Star (Bottom)"

    return None

def evening_star_top(df):
    if len(df) < 60:
        return None

    ema50 = talib.EMA(df["close"], 50)

    if df["close"].iloc[-1] < ema50.iloc[-1]:
        return None

    pattern = talib.CDLEVENINGSTAR(
        df["open"], df["high"], df["low"], df["close"]
    ).iloc[-1]

    if pattern < 0:
        return "Evening Star (Top)"

    return None

def bullish_gsas(df_tf, df_htf):
    rsi = talib.RSI(df_tf["close"], 14)
    adx = talib.ADX(df_tf["high"], df_tf["low"], df_tf["close"], 14)
    ubb, _, _ = talib.BBANDS(df_tf["close"], 20)

    macd_htf, sig_htf, _ = talib.MACD(df_htf["close"], 12, 26, 9)
    ema20_htf = talib.EMA(df_htf["close"], 20)

    if (
        rsi.iloc[-1] > 60 and
        ubb.iloc[-1] > ubb.iloc[-2] and
        adx.iloc[-1] > adx.iloc[-2] and adx.iloc[-2] < adx.iloc[-3] and
        macd_htf.iloc[-1] > sig_htf.iloc[-1] and
        df_htf["close"].iloc[-1] > ema20_htf.iloc[-1]
    ):
        return "Bullish GSAS"

    return None


def bearish_gsas(df_tf, df_htf):
    rsi = talib.RSI(df_tf["close"], 14)
    adx = talib.ADX(df_tf["high"], df_tf["low"], df_tf["close"], 14)
    _, _, lbb = talib.BBANDS(df_tf["close"], 20)

    macd_htf, sig_htf, _ = talib.MACD(df_htf["close"], 12, 26, 9)
    ema20_htf = talib.EMA(df_htf["close"], 20)

    if (
        rsi.iloc[-1] < 60 and
        lbb.iloc[-1] < lbb.iloc[-2] and
        adx.iloc[-1] > adx.iloc[-2] and adx.iloc[-2] < adx.iloc[-3] and
        macd_htf.iloc[-1] < sig_htf.iloc[-1] and
        df_htf["close"].iloc[-1] < ema20_htf.iloc[-1]
    ):
        return "Bearish GSAS"

    return None

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
        "Probable C Wave",
        "MACD Bearish Peak Divergence",
        "MACD Bullish Base Divergence",
        "Trend Alignment (EMA)",
        "Pullback to EMA",
        "High Probability Confluence",
        "MACD Hook Up",
        "MACD Hook Down",
        "MACD Histogram Divergence",
        "EMA50 + Stoch Oversold",
        "Dark Cloud Cover",
        "Morning Star (Bottom)",
        "Evening Star (Top)",
        "Bullish GSAS",
        "Bearish GSAS",
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

    # === Higher Timeframe preload for GSAS ===
    if scanner in ["Bullish GSAS", "Bearish GSAS"]:
        htf_map = {
            "15 Min": "1 Hour",
            "1 Hour": "Daily",
            "Daily": "Weekly",
        }

        if tf not in htf_map:
            st.warning("GSAS not supported for this timeframe")
            st.stop()

        data_htf = load_data(TIMEFRAMES[htf_map[tf]])
    else:
        data_htf = None

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
                
        elif scanner == "Probable C Wave":
            if c_wave_finder(df):
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
                results.append({"Symbol": sym, "Trend": sig})

        elif scanner == "Pullback to EMA":
            sig = pullback_to_ema(df)
            if sig:
                results.append({"Symbol": sym, "Setup": sig})

        elif scanner == "High Probability Confluence":
            sig = confluence_setup(df)
            if sig:
                results.append({"Symbol": sym, "Setup": sig})

        elif scanner == "MACD Hook Up":
            sig = macd_hook_up(df)
            if sig:
                results.append({"Symbol": sym, "Signal": sig})

        elif scanner == "MACD Hook Down":
            sig = macd_hook_down(df)
            if sig:
                results.append({"Symbol": sym, "Signal": sig})

        elif scanner == "MACD Histogram Divergence":
            sig = macd_histogram_divergence(df)
            if sig:
                results.append({"Symbol": sym, "Signal": sig})

        elif scanner == "EMA50 + Stoch Oversold":
            sig = ema50_stoch_oversold(df)
            if sig:
                results.append({"Symbol": sym, "Signal": sig})

        elif scanner == "Dark Cloud Cover":
            sig = dark_cloud_cover(df)
            if sig:
                results.append({"Symbol": sym, "Signal": sig})

        elif scanner == "Morning Star (Bottom)":
            sig = morning_star_bottom(df)
            if sig:
                results.append({"Symbol": sym, "Signal": sig})

        elif scanner == "Evening Star (Top)":
            sig = evening_star_top(df)
            if sig:
                results.append({"Symbol": sym, "Signal": sig})

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

        elif scanner == "Bullish GSAS":
            if data_htf is not None and sym in data_htf:
                sig = bullish_gsas(df, data_htf[sym])
                if sig:
                    results.append({"Symbol": sym, "Signal": sig})

        elif scanner == "Bearish GSAS":
            if data_htf is not None and sym in data_htf:
                sig = bearish_gsas(df, data_htf[sym])
                if sig:
                    results.append({"Symbol": sym, "Signal": sig})

    if not results:
        st.info("No stocks matched.")
        st.stop()

    df_res = pd.DataFrame(results)
    df_res = df_res.replace([float("inf"), float("-inf")], "")
    df_res = df_res.fillna("")
    for c in df_res.columns:
        df_res[c] = df_res[c].astype(str)
    st.success(f"Stocks Found: {len(df_res)}")
    st.dataframe(df_res, use_container_width=True)



    if scanner == "RSI Market Pulse":
        fig = px.histogram(df_res, x="Zone", title="RSI Zone Distribution")
        st.plotly_chart(fig, use_container_width=True)


st.markdown("""
---
**Designed by:-  
Gaurav Singh Yadav**   
🩷💛🩵💙🩶💜🤍🤎💖  Built With Love 🫶  
Energy | Commodity | Quant Intelligence 📶  
📱 +91-8003994518 〽️   
📧 yadav.gauravsingh@gmail.com ™️
""")
