import streamlit as st
import os
import pandas as pd
import numpy as np
import sys
if sys.version_info >= (3, 13):
    import types
    imghdr = types.ModuleType("imghdr")
    imghdr.what = lambda *args, **kwargs: None
    sys.modules["imghdr"] = imghdr

import talib
import plotly.express as px
import hashlib
import base64

def set_bg_image(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )



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
set_bg_image("Assets/BG1.jpeg")

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
        st.session_state.clear()
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
# BACKTEST DATE SELECTION
# ==================================================
st.sidebar.markdown("### 📅 Backtest Date")

analysis_date = st.sidebar.date_input(
    "Select Analysis Date",
    value=last_d.date() if last_d else pd.Timestamp.today().date()
)

st.sidebar.info(
    f"Backtest Mode Active\nData cutoff: {analysis_date}"
)


st.sidebar.caption(
    f"Scans will run as of: {analysis_date}"
)

def trim_df_to_date(df, anchor_date):
    """
    Cuts dataframe so that the last candle <= anchor_date
    """
    if df is None or df.empty:
        return None

    df = df.copy()

    if isinstance(df.index, pd.DatetimeIndex):
        df = df[df.index.date <= anchor_date]
    elif "datetime" in df.columns:
        df = df[df["datetime"].dt.date <= anchor_date]

    if len(df) < 120:
        return None

    return df






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
        if curr["open"] < prev["close"] and curr["close"] >= mid:
            return "Bullish"
    if prev["close"] > prev["open"] and curr["close"] < curr["open"]:
        if curr["open"] > prev["close"] and curr["close"] <= mid:
            return "Bearish"
    return None

# --- Breakaway Gaps
def breakaway_gap(df):
    if len(df) < 50:
        return None
    df = df.copy()
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


def macd_rd(df, df_htf):
    if len(df) < 60 or len(df_htf) < 30:
        return None

    # --- LTF MACD ---
    macd, signal, _ = talib.MACD(df["close"], 12, 26, 9)
    latest = macd.iloc[-1]
    prev = macd.iloc[-2]
    sig = signal.iloc[-1]

    max60 = macd.rolling(60).max().iloc[-1]

    # --- HTF MACD ---
    macd_htf, _, _ = talib.MACD(df_htf["close"], 12, 26, 9)
    macd_htf_val = macd_htf.iloc[-1]

    if (
        latest > prev and
        latest > 0 and
        sig < latest and              # ✅ MACD > Signal
        macd_htf_val > 0 and           # ✅ HTF trend filter
        max60 > 0 and
        (latest / max60) < 0.25
    ):
        return "MACD RD (Compression + Trend Aligned)"

    return None



def third_wave_finder(df):
    if len(df) < 100:
        return False

    # EMA calculation
    ema20 = talib.EMA(df["close"], 20)
    ema50 = talib.EMA(df["close"], 50)

    # 1️⃣ Bullish EMA 20 / EMA 50 crossover (PCO)
    if not (
        ema20.iloc[-1] > ema50.iloc[-1] and
        ema20.iloc[-2] < ema50.iloc[-2]
    ):
        return False

    # 2️⃣ Find LOW before the crossover (Wave-2 low)
    # Using candles before the crossover zone
    pre_crossover_low = df["low"].iloc[-60:-30].min()

    # 3️⃣ Find HIGH after crossover (Wave-1 high / pivot high)
    post_crossover_high = df["high"].iloc[-30:].max()

    # Safety check
    if post_crossover_high <= pre_crossover_low:
        return False

    # 4️⃣ Measure impulse move
    move = post_crossover_high - pre_crossover_low

    # 5️⃣ 50% retracement level
    retrace_50 = pre_crossover_low + (move * 0.5)

    # 6️⃣ Check if price retraced near 50% (Wave-2 pullback)
    current_price = df["close"].iloc[-1]

    if retrace_50 * 0.95 <= current_price <= retrace_50 * 1.05:
        return True

    return False


def c_wave_finder(df):
    if len(df) < 100:
        return False

    # EMA calculation
    ema20 = talib.EMA(df["close"], 20)
    ema50 = talib.EMA(df["close"], 50)

    # 1️⃣ Bearish EMA 20 / EMA 50 crossover
    if not (
        ema20.iloc[-1] < ema50.iloc[-1] and
        ema20.iloc[-2] > ema50.iloc[-2]
    ):
        return False

    # 2️⃣ Find HIGH before crossover (Wave-B high)
    pre_crossover_high = df["high"].iloc[-60:-30].max()

    # 3️⃣ Find LOW after crossover (Wave-A low)
    post_crossover_low = df["low"].iloc[-30:].min()

    # Safety check
    if post_crossover_low >= pre_crossover_high:
        return False

    # 4️⃣ Measure impulse move
    move = pre_crossover_high - post_crossover_low

    # 5️⃣ 50% retracement level (pullback upward)
    retrace_50 = post_crossover_low + (move * 0.5)

    # 6️⃣ Check if price retraced near 50%
    current_price = df["close"].iloc[-1]

    if retrace_50 * 0.95 <= current_price <= retrace_50 * 1.05:
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

# --- 50 EMA Fake Breakdown (Bullish Trap below EMA50)
def ema50_fake_breakdown(df):
    if len(df) < 55:
        return None

    df = df.copy()
    df["EMA20"] = talib.EMA(df["close"], 20)
    df["EMA50"] = talib.EMA(df["close"], 50)

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    # Condition:
    # Current close > EMA50
    # Previous close < EMA50
    # EMA20 > EMA50 (bullish structure)
    if (
        curr["close"] > curr["EMA50"] and
        prev["close"] < prev["EMA50"] and
        curr["EMA20"] > curr["EMA50"]
    ):
        return "50 EMA Fake Breakdown"

    return None


# --- 50 EMA Fake Breakout (Bearish Trap above EMA50)
def ema50_fake_breakout(df):
    if len(df) < 55:
        return None

    df = df.copy()
    df["EMA20"] = talib.EMA(df["close"], 20)
    df["EMA50"] = talib.EMA(df["close"], 50)

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    # Condition:
    # Current close < EMA50
    # Previous close > EMA50
    # EMA20 < EMA50 (bearish structure)
    if (
        curr["close"] < curr["EMA50"] and
        prev["close"] > prev["EMA50"] and
        curr["EMA20"] < curr["EMA50"]
    ):
        return "50 EMA Fake Breakout"

    return None


# ===============================
# KDJ CALCULATION (Pine → Python)
# ===============================
def kdj(df, period=9, signal=3):
    """
    Returns pK, pD, pJ series
    """
    low_min  = df["low"].rolling(period).min()
    high_max = df["high"].rolling(period).max()

    rsv = 100 * (df["close"] - low_min) / (high_max - low_min)

    # bcwsma equivalent
    def bcwsma(series, length, m=1):
        out = []
        for i, val in enumerate(series):
            if i == 0 or np.isnan(val):
                out.append(val)
            else:
                out.append((m * val + (length - m) * out[i-1]) / length)
        return pd.Series(out, index=series.index)

    pK = bcwsma(rsv, signal, 1)
    pD = bcwsma(pK, signal, 1)
    pJ = 3 * pK - 2 * pD

    return pK, pD, pJ

def kdj_buy(df):
    if len(df) < 15:
        return None

    pK, pD, pJ = kdj(df)

    # Cross condition
    crossed_up = (
        pJ.iloc[-2] < pD.iloc[-2] and
        pJ.iloc[-1] > pD.iloc[-1]
    )

    oversold = pD.iloc[-1] < 20 and pJ.iloc[-1] < 20

    if crossed_up and oversold:
        return "KDJ BUY (J↑D below 20)"

    return None

def kdj_sell(df):
    if len(df) < 15:
        return None

    pK, pD, pJ = kdj(df)

    crossed_down = (
        pJ.iloc[-2] > pD.iloc[-2] and
        pJ.iloc[-1] < pD.iloc[-1]
    )

    overbought = pD.iloc[-1] > 80 and pJ.iloc[-1] > 80

    if crossed_down and overbought:
        return "KDJ SELL (J↓D above 80)"

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
        "50 EMA Fake Breakdown",
        "50 EMA Fake Breakout",
        "KDJ BUY (Oversold)",
        "KDJ SELL (Overbought)",

    ]
)

run = st.sidebar.button("▶ Run Scan")

# ==================================================
# MAIN EXECUTION
# ==================================================
df_res = pd.DataFrame()  # 👈 SAFE DEFAULT
if run:
    data = load_data(TIMEFRAMES[tf])
    if not data:
        st.warning("No data found.")
        st.stop()

    results = []

    # === Higher Timeframe preload for GSAS ===
    if scanner in ["Bullish GSAS", "Bearish GSAS", "MACD RD (4th Wave)"]:

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
        df = trim_df_to_date(df, analysis_date)
        if df is None:
            continue

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
            if data_htf is not None and sym in data_htf:
                df_htf = trim_df_to_date(data_htf[sym], analysis_date)
                if df_htf is None:
                    continue
                sig = macd_rd(df, df_htf)
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
                df_wt = trim_df_to_date(data_w[sym], analysis_date)
                df_mt = trim_df_to_date(data_m[sym], analysis_date)
                
                if df_wt is None or df_mt is None:
                    continue
                sig = rsi_wm(df, df_wt, df_mt)
                if sig:
                    results.append({"Symbol": sym, "Signal": sig})

        elif scanner == "Bullish GSAS":
            if data_htf is not None and sym in data_htf:
                df_htf = trim_df_to_date(data_htf[sym], analysis_date)
                if df_htf is None:
                    continue
                sig = bullish_gsas(df, df_htf)
                if sig:
                    results.append({"Symbol": sym, "Signal": sig})

        elif scanner == "Bearish GSAS":
            if data_htf is not None and sym in data_htf:
                df_htf = trim_df_to_date(data_htf[sym], analysis_date)
                if df_htf is None:
                    continue
                sig = bearish_gsas(df, df_htf)

                if sig:
                    results.append({"Symbol": sym, "Signal": sig})

        elif scanner == "50 EMA Fake Breakdown":
            sig = ema50_fake_breakdown(df)
            if sig:
                results.append({"Symbol": sym, "Signal": sig})

        elif scanner == "50 EMA Fake Breakout":
            sig = ema50_fake_breakout(df)
            if sig:
                results.append({"Symbol": sym, "Signal": sig})

        elif scanner == "KDJ BUY (Oversold)":
            sig = kdj_buy(df)
            if sig:
                results.append({"Symbol": sym, "Signal": sig})

        elif scanner == "KDJ SELL (Overbought)":
            sig = kdj_sell(df)
            if sig:
                results.append({"Symbol": sym, "Signal": sig})


    if not results:
        st.info("No stocks matched.")
        df_res = pd.DataFrame()
    else:
        df_res = pd.DataFrame(results)


    df_res = pd.DataFrame(results)

    ALL_COLS = ["Symbol", "Signal", "Trend", "State", "Setup", "Divergence", "RSI", "Zone"]
    for c in ALL_COLS:
        if c not in df_res.columns:
            df_res[c] = ""

    df_res = df_res[ALL_COLS]
    df_res = df_res.replace([np.inf, -np.inf], "")
    df_res = df_res.fillna("")

    # 🔹 Use numeric df_res for charts
    # 🔹 Create string-only copy for table
    df_display = df_res.copy().astype(str)

    st.dataframe(df_display, use_container_width=True)



# =============================
# MARKET PULSE DASHBOARD AREA
# =============================
pulse_container = st.container()
with pulse_container:
    if scanner == "RSI Market Pulse" and not df_res.empty:

        colA, colB = st.columns(2)

        # --- RSI Zones count (एक बार ही) ---
        bull = (df_res["Zone"] == "RSI > 60").sum()
        neutral = (df_res["Zone"] == "RSI 40–60").sum()
        bear = (df_res["Zone"] == "RSI < 40").sum()

        # --- Donut Chart ---
        df_pie = pd.DataFrame({
            "Zone": ["RSI > 60", "RSI 40–60", "RSI < 40"],
            "Count": [bull, neutral, bear]
        })

        fig = px.pie(
            df_pie,
            names="Zone",
            values="Count",
            hole=0.55,
            color="Zone",
            color_discrete_map={
                "RSI > 60": "#2ecc71",
                "RSI 40–60": "#deaf68",
                "RSI < 40": "#e74c3c",
            },
            title="📊 RSI Market Breadth"
        )
        # यहां percent + label दोनों दिखेंगे
        fig.update_traces(
            textinfo="percent+label",
            texttemplate="%{percent:.1%} %{label}"
        )


        colA.plotly_chart(fig, use_container_width=True)

        # --- Breadth Metrics ---
        zone_total = bull + neutral + bear

        if zone_total == 0:
            st.warning("No valid RSI zones for sentiment calculation")
            st.stop()

        bull_pct = (bull / zone_total) * 100
        bear_pct = (bear / zone_total) * 100
        neutral_pct = (neutral / zone_total) * 100



        msi = (bull - bear) / zone_total
        msi_pct = msi * 100

        if msi > 0.2:
            delta = msi_pct
            delta_color = "normal"
        elif msi < -0.2:
            delta = msi_pct
            delta_color = "inverse"
        else:
            delta = 0
            delta_color = "off"

        colB.metric("🟢 Bullish Strength", f"{bull_pct:.2f}%")

        bear_delta = -bear_pct
        colB.metric(
            "🔴 Bearish Weakness",
            f"{bear_pct:.2f}%",
            delta=bear_delta,
            delta_color="inverse"
        )

        colB.metric("⚖️ Neutral", f"{neutral_pct:.2f}%")
        colB.metric(
            "🧠 Market Sentiment Index",
            f"{msi_pct:.2f}%",
            delta=delta,
            delta_color=delta_color
        )






with pulse_container:
    if scanner == "MACD Market Pulse" and not df_res.empty:

        st.markdown("### 📈 MACD Trend Strength Overview")

        state_counts = (
            df_res["State"]
            .replace("", "No Signal")
            .value_counts()
            .reset_index()
        )
        state_counts.columns = ["State", "Count"]

        fig_macd = px.bar(
            state_counts,
            x="Count",
            y="State",
            orientation="h",
            title="MACD Trend States Distribution",
            text="Count"
        )

        st.plotly_chart(fig_macd, use_container_width=True, key="macd_bar")

        col1, col2 = st.columns(2)

        strong_bull = state_counts[state_counts["State"] == "Strong Bullish"]["Count"].sum()
        strong_bear = state_counts[state_counts["State"] == "Strong Bearish"]["Count"].sum()

        col1.metric("🚀 Strong Bullish Stocks", strong_bull)
        col2.metric("🧨 Strong Bearish Stocks", strong_bear)


    


st.markdown("""
---
**Designed by:-  
Gaurav Singh Yadav**   
🩷💛🩵💙🩶💜🤍🤎💖  Built With Love 🫶  
Energy | Commodity | Quant Intelligence 📶  
📱 +91-8003994518 〽️   
📧 yadav.gauravsingh@gmail.com ™️
""")
