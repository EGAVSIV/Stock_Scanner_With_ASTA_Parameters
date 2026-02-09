import os
import sys
import base64
import hashlib

import numpy as np
import pandas as pd
import streamlit as st
import talib
import plotly.express as px

# --- Python 3.13 image hack (as you had) ---
if sys.version_info >= (3, 13):
    import types

    imghdr = types.ModuleType("imghdr")
    imghdr.what = lambda *args, **kwargs: None
    sys.modules["imghdr"] = imghdr

# ==============================
# GLOBAL CONFIG
# ==============================
SAFE_COLS = [
    "Symbol",
    "Signal",
    "Trend",
    "State",
    "Setup",
    "Divergence",
    "RSI",
    "Zone",
    "Confluence",
    "Bias",
    "Probability",
    "TV_Link",
]

BULL_KEYWORDS = ["Bullish", "BUY", "Breakout", "Uptrend", "Momentum"]
BEAR_KEYWORDS = ["Bearish", "SELL", "Breakdown", "Downtrend"]


def empty_result_df():
    return pd.DataFrame({c: [] for c in SAFE_COLS})


def set_bg_image(image_path: str):
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
        unsafe_allow_html=True,
    )


def hash_pwd(pwd: str) -> str:
    return hashlib.sha256(pwd.encode()).hexdigest()


# ==============================
# AUTH
# ==============================
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

# ==============================
# LAST CANDLE TIME HELPERS
# ==============================
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


# ==============================
# STREAMLIT CONFIG
# ==============================
st.set_page_config("Rao_G", layout="wide", page_icon="🧮")

# Background
set_bg_image("Assets/BG1.jpeg")

# Global styling (Inter + glass)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #020617;
    }

    .main-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 32px;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #f1f2f6;
        text-shadow: 0 0 16px rgba(0,0,0,0.9);
    }

    .subtitle {
        font-size: 14px;
        color: #ced6e0;
        opacity: 0.9;
    }

    .glass-card {
        background: rgba(15, 15, 15, 0.82);
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 14px 35px rgba(0,0,0,0.6);
        padding: 16px 18px;
        backdrop-filter: blur(18px);
    }

    .scanner-chip {
        border-radius: 999px;
        padding: 10px 14px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.18);
        box-shadow: 0 6px 16px rgba(0,0,0,0.45);
        cursor: pointer;
        transition: all 0.18s ease-out;
    }

    .scanner-chip:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 10px 24px rgba(0,0,0,0.55);
        filter: brightness(1.08);
    }

    .scanner-chip span {
        font-weight: 600;
        font-size: 13px;
        color: #ffffff;
        letter-spacing: 0.02em;
    }

    .scanner-chip-active {
        border: 2px solid #ffffff;
    }

    table {
        border-collapse: collapse;
        width: 100%;
        font-size: 13px;
        color: #ecf0f1;
    }

    thead tr {
        background: linear-gradient(90deg, #1e272e, #2f3640);
    }

    th, td {
        border: 1px solid rgba(255,255,255,0.06);
        padding: 6px 10px;
    }

    tbody tr:nth-child(even) {
        background: rgba(20,20,20,0.78);
    }

    tbody tr:nth-child(odd) {
        background: rgba(10,10,10,0.80);
    }

    tbody tr:hover {
        background: rgba(39, 60, 117, 0.75);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827, #020617);
        border-right: 1px solid rgba(148,163,184,0.3);
    }

    .pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border-radius: 999px;
        background: rgba(15,23,42,0.9);
        border: 1px solid rgba(148,163,184,0.6);
        padding: 4px 10px;
        font-size: 11px;
        color: #e5e7eb;
    }

    .pill-dot {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: #22c55e;
        box-shadow: 0 0 8px rgba(34,197,94,0.9);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title block
st.markdown(
    """
    <div class="glass-card" style="margin-bottom: 0.8rem;">
        <div class="main-title">Multi-Timeframe Stock Screener</div>
        <div class="subtitle">
            Quant-powered scans • RSI • MACD • GSAS • Patterns • Multi-timeframe confluence
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 1) TIMEFRAMES
TIMEFRAMES = {
    "15 Min": "stock_data_15",
    "1 Hour": "stock_data_1H",
    "Daily": "stock_data_D",
    "Weekly": "stock_data_W",
    "Monthly": "stock_data_M",
}

# 2) Sidebar: timeframe
tf_options = list(TIMEFRAMES.keys())
tf = st.sidebar.selectbox("Timeframe", tf_options, index=tf_options.index("Daily"))

# 3) DATA LOADER
@st.cache_data(show_spinner=False)
def load_data(folder: str):
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


def make_tradingview_link(sym: str) -> str:
    base = "https://in.tradingview.com/chart/LqUZraZ9/"
    return f"{base}?symbol=NSE%3A{sym}"


# Single stock dropdown
sample_data = load_data(TIMEFRAMES[tf])
all_symbols = sorted(sample_data.keys()) if sample_data else []
st.sidebar.markdown("### 🔍 Single Stock Scan")
selected_symbol = st.sidebar.selectbox(
    "Select Stock (for current timeframe)",
    all_symbols if all_symbols else ["NA"],
)

# Last candle info
last_15m = get_last_candle_by_tf(TIMEFRAMES["15 Min"])
last_1h = get_last_candle_by_tf(TIMEFRAMES["1 Hour"])
last_d = get_last_candle_by_tf(TIMEFRAMES["Daily"])
last_w = get_last_candle_by_tf(TIMEFRAMES["Weekly"])
last_m = get_last_candle_by_tf(TIMEFRAMES["Monthly"])

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
        unsafe_allow_html=False,
    )

st.markdown("---")

# ==============================
# BACKTEST DATE
# ==============================
st.sidebar.markdown("### 📅 Backtest Date")
analysis_date = st.sidebar.date_input(
    "Select Analysis Date",
    value=last_d.date() if last_d else pd.Timestamp.today().date(),
)
st.sidebar.info(f"Backtest Mode Active\nData cutoff: {analysis_date}")
st.sidebar.caption(f"Scans will run as of: {analysis_date}")


def trim_df_to_date(df: pd.DataFrame, anchor_date):
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


# ==============================
# SCANNERS (PURE FUNCTIONS)
# ==============================
# --- all your scanner functions exactly as in your file ---
# (rsi_market_pulse, volume_shocker, nrb_7, counter_attack, breakaway_gap,
#  rsi_adx, rsi_wm, macd_market_pulse, macd_normal_divergence, macd_rd,
#  third_wave_finder, c_wave_finder, macd_peak_bearish_divergence,
#  macd_base_bullish_divergence, trend_alignment, pullback_to_ema,
#  confluence_setup, macd_hook_up, macd_hook_down, macd_histogram_divergence,
#  ema50_stoch_oversold, dark_cloud_cover, morning_star_bottom,
#  evening_star_top, bullish_gsas, bearish_gsas, ema50_fake_breakdown,
#  ema50_fake_breakout, kdj, kdj_buy, kdj_sell, consecutive_close_momentum,
#  camarilla_breakout, cpr_breakout, inside_bar_breakout, adx_expansion,
#  range_expansion_day, failed_breakout_breakdown,
#  ema_compression_expansion, atr_percent)
# >>> यहाँ पर वही functions paste रखें जो तुम्हारे latest code में हैं (मैंने ऊपर से कॉपी नहीं छेड़ा) <<<


# ==============================
# CONFLUENCE CALC
# ==============================
def calculate_confluence(row):
    score = 0
    text = " ".join(
        [
            str(row.get("Signal", "")),
            str(row.get("Trend", "")),
            str(row.get("State", "")),
            str(row.get("Setup", "")),
            str(row.get("Divergence", "")),
        ]
    )

    for k in BULL_KEYWORDS:
        if k in text:
            score += 1

    for k in BEAR_KEYWORDS:
        if k in text:
            score -= 1

    score = max(min(score, 5), -5)

    if score > 0:
        bias = "Bullish"
    elif score < 0:
        bias = "Bearish"
    else:
        bias = "Neutral"

    abs_score = abs(score)

    if abs_score >= 4:
        prob = "High"
    elif abs_score >= 3:
        prob = "Medium"
    else:
        prob = "Low"

    return score, bias, prob


# ==============================
# RUN ALL SCANNERS FOR SINGLE SYMBOL
# ==============================
def run_all_scanners_for_symbol(
    sym,
    df,
    tf,
    analysis_date,
    data_all_tfs,
):
    results = {}

    # Basic single-TF scanners
    results["RSI Market Pulse"] = rsi_market_pulse(df) is not None
    results["Volume Shocker"] = volume_shocker(df)
    results["NRB-7 Breakout"] = nrb_7(df) is not None
    results["Counter Attack"] = counter_attack(df) is not None
    results["Breakaway Gaps"] = breakaway_gap(df) is not None
    results["RSI + ADX"] = rsi_adx(df) is not None
    results["MACD Market Pulse"] = macd_market_pulse(df) is not None
    results["MACD Normal Divergence"] = macd_normal_divergence(df) is not None

    results["MACD Bearish Peak Divergence"] = (
        macd_peak_bearish_divergence(df) is not None
    )
    results["MACD Bullish Base Divergence"] = (
        macd_base_bullish_divergence(df) is not None
    )
    results["Trend Alignment (EMA)"] = trend_alignment(df) is not None
    results["Pullback to EMA"] = pullback_to_ema(df) is not None
    results["High Probability Confluence"] = confluence_setup(df) is not None
    results["MACD Hook Up"] = macd_hook_up(df) is not None
    results["MACD Hook Down"] = macd_hook_down(df) is not None
    results["MACD Histogram Divergence"] = (
        macd_histogram_divergence(df) is not None
    )
    results["EMA50 + Stoch Oversold"] = ema50_stoch_oversold(df) is not None
    results["Dark Cloud Cover"] = dark_cloud_cover(df) is not None
    results["Morning Star (Bottom)"] = morning_star_bottom(df) is not None
    results["Evening Star (Top)"] = evening_star_top(df) is not None
    results["50 EMA Fake Breakdown"] = ema50_fake_breakdown(df) is not None
    results["50 EMA Fake Breakout"] = ema50_fake_breakout(df) is not None
    results["KDJ BUY (Oversold)"] = kdj_buy(df) is not None
    results["KDJ SELL (Overbought)"] = kdj_sell(df) is not None
    results["Probable Momentum (Consecutive Close)"] = (
        consecutive_close_momentum(df, min_count=3) is not None
    )
    results["Camarilla Breakout / Breakdown"] = (
        camarilla_breakout(df) is not None
    )
    results["CPR Breakout / Breakdown"] = cpr_breakout(df) is not None
    results["Inside Bar Breakout"] = inside_bar_breakout(df) is not None
    results["ADX Expansion (Trend Ignition)"] = adx_expansion(df) is not None
    results["Range Expansion Day"] = range_expansion_day(df) is not None
    results["Failed Breakout / Breakdown"] = (
        failed_breakout_breakdown(df) is not None
    )
    results["EMA Compression → Expansion"] = (
        ema_compression_expansion(df) is not None
    )

    # Multi-TF
    if "Weekly" in data_all_tfs and "Monthly" in data_all_tfs:
        data_w = data_all_tfs["Weekly"]
        data_m = data_all_tfs["Monthly"]
        if sym in data_w and sym in data_m:
            df_w = trim_df_to_date(data_w[sym], analysis_date)
            df_m = trim_df_to_date(data_m[sym], analysis_date)
            if df_w is not None and df_m is not None:
                results["RSI WM 60–40"] = (
                    rsi_wm(df, df_w, df_m) is not None
                )
            else:
                results["RSI WM 60–40"] = False
        else:
            results["RSI WM 60–40"] = False
    else:
        results["RSI WM 60–40"] = False

    htf_map = {
        "15 Min": "1 Hour",
        "1 Hour": "Daily",
        "Daily": "Weekly",
        "Weekly": "Monthly",
    }
    if tf in htf_map and htf_map[tf] in data_all_tfs:
        data_htf = data_all_tfs[htf_map[tf]]
    else:
        data_htf = None

    if data_htf is not None and sym in data_htf:
        df_htf = trim_df_to_date(data_htf[sym], analysis_date)
        if df_htf is not None:
            results["MACD RD (4th Wave)"] = macd_rd(df, df_htf) is not None
        else:
            results["MACD RD (4th Wave)"] = False
    else:
        results["MACD RD (4th Wave)"] = False

    results["Probable 3rd Wave"] = third_wave_finder(df)
    results["Probable C Wave"] = c_wave_finder(df)

    if data_htf is not None and sym in data_htf:
        df_htf = trim_df_to_date(data_htf[sym], analysis_date)
        if df_htf is not None:
            results["Bullish GSAS"] = bullish_gsas(df, df_htf) is not None
            results["Bearish GSAS"] = bearish_gsas(df, df_htf) is not None
        else:
            results["Bullish GSAS"] = False
            results["Bearish GSAS"] = False
    else:
        results["Bullish GSAS"] = False
        results["Bearish GSAS"] = False

    atr_val = atr_percent(df)
    results["Top 10 by ATR %"] = atr_val is not None

    return results


# ==============================
# SCANNER TILE CONFIG (grouped + sorted)
# ==============================
SCANNERS = [
    {"name": "ADX Expansion (Trend Ignition)", "color": "#e67e22", "group": "Trend & Volatility"},
    {"name": "Breakaway Gaps", "color": "#e67e22", "group": "Gap & Range"},
    {"name": "Bullish GSAS", "color": "#27ae60", "group": "Multi-TF GSAS"},
    {"name": "Bearish GSAS", "color": "#27ae60", "group": "Multi-TF GSAS"},
    {"name": "Camarilla Breakout / Breakdown", "color": "#f39c12", "group": "Pivot & Levels"},
    {"name": "Counter Attack", "color": "#1abc9c", "group": "Candlestick"},
    {"name": "CPR Breakout / Breakdown", "color": "#e67e22", "group": "Pivot & Levels"},
    {"name": "Dark Cloud Cover", "color": "#f1c40f", "group": "Candlestick"},
    {"name": "EMA Compression → Expansion", "color": "#34495e", "group": "EMA Structure"},
    {"name": "EMA50 + Stoch Oversold", "color": "#f1c40f", "group": "EMA + Oscillators"},
    {"name": "Evening Star (Top)", "color": "#f1c40f", "group": "Candlestick"},
    {"name": "Failed Breakout / Breakdown", "color": "#34495e", "group": "Reversal / Failure"},
    {"name": "High Probability Confluence", "color": "#e84393", "group": "Confluence"},
    {"name": "Inside Bar Breakout", "color": "#e67e22", "group": "Range / Inside Bar"},
    {"name": "KDJ BUY (Oversold)", "color": "#f39c12", "group": "KDJ"},
    {"name": "KDJ SELL (Overbought)", "color": "#f39c12", "group": "KDJ"},
    {"name": "MACD Bearish Peak Divergence", "color": "#3498db", "group": "MACD Divergence"},
    {"name": "MACD Bullish Base Divergence", "color": "#3498db", "group": "MACD Divergence"},
    {"name": "MACD Histogram Divergence", "color": "#e84393", "group": "MACD Divergence"},
    {"name": "MACD Hook Down", "color": "#e84393", "group": "MACD Hooks"},
    {"name": "MACD Hook Up", "color": "#e84393", "group": "MACD Hooks"},
    {"name": "MACD Market Pulse", "color": "#e67e22", "group": "MACD"},
    {"name": "MACD Normal Divergence", "color": "#f1c40f", "group": "MACD Divergence"},
    {"name": "MACD RD (4th Wave)", "color": "#f1c40f", "group": "MACD Multi-TF"},
    {"name": "Morning Star (Bottom)", "color": "#f1c40f", "group": "Candlestick"},
    {"name": "NRB-7 Breakout", "color": "#1abc9c", "group": "Range / Inside Bar"},
    {"name": "Probable 3rd Wave", "color": "#f1c40f", "group": "Wave Structure"},
    {"name": "Probable C Wave", "color": "#f1c40f", "group": "Wave Structure"},
    {"name": "Probable Momentum (Consecutive Close)", "color": "#f39c12", "group": "Momentum"},
    {"name": "Range Expansion Day", "color": "#e67e22", "group": "Trend & Volatility"},
    {"name": "RSI + ADX", "color": "#e67e22", "group": "RSI + ADX"},
    {"name": "RSI Market Pulse", "color": "#1abc9c", "group": "RSI"},
    {"name": "RSI WM 60–40", "color": "#e67e22", "group": "RSI Multi-TF"},
    {"name": "Top 10 by ATR %", "color": "#9b59b6", "group": "ATR"},
    {"name": "Trend Alignment (EMA)", "color": "#3498db", "group": "EMA Structure"},
    {"name": "Volume Shocker", "color": "#1abc9c", "group": "Volume"},
    {"name": "50 EMA Fake Breakdown", "color": "#27ae60", "group": "EMA Traps"},
    {"name": "50 EMA Fake Breakout", "color": "#27ae60", "group": "EMA Traps"},
]

SCANNERS = sorted(SCANNERS, key=lambda x: x["name"])

if "scanner" not in st.session_state:
    st.session_state["scanner"] = SCANNERS[0]["name"]

st.markdown("### 🎯 Scanner Palette")

cols_per_row = 4
clicked_scanner = None

groups = {}
for sc in SCANNERS:
    groups.setdefault(sc["group"], []).append(sc)

for group_name, scanners_in_group in groups.items():
    st.markdown(
        f"<div class='glass-card' style='margin-bottom: 0.5rem;'>"
        f"<div style='font-size:13px; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:#9ca3af;'>{group_name}</div>",
        unsafe_allow_html=True,
    )
    for i in range(0, len(scanners_in_group), cols_per_row):
        row = scanners_in_group[i : i + cols_per_row]
        cols = st.columns(len(row))
        for col, sc in zip(cols, row):
            with col:
                is_active = st.session_state["scanner"] == sc["name"]
                extra_class = "scanner-chip-active" if is_active else ""
                st.markdown(
                    f"""
                    <div class="scanner-chip {extra_class}" style="background:{sc['color']}; margin-top:6px;">
                        <span>{sc["name"]}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(f"Scan: {sc['name']}", key=f"btn_{sc['name']}"):
                    clicked_scanner = sc["name"]
    st.markdown("</div>", unsafe_allow_html=True)

if clicked_scanner is not None:
    st.session_state["scanner"] = clicked_scanner

scanner = st.session_state["scanner"]

st.markdown(
    f"""
    <div style="margin-top:8px; margin-bottom:4px;">
        <span class="pill-badge">
            <span class="pill-dot"></span>
            <span style="text-transform:uppercase; font-size:10px; opacity:0.75;">ACTIVE</span>
            <span style="font-weight:600;">{scanner}</span>
            <span style="opacity:0.55;">•</span>
            <span style="opacity:0.8;">TF: {tf}</span>
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

run = clicked_scanner is not None

df_res = empty_result_df()

# ==============================
# MAIN EXECUTION (तुम्हारा existing logic as-is रखें)
# ==============================
if run:
    # यहाँ तुम्हारा पहले से existing main scanning loop होगा
    # (data = load_data(...), results list, atr_list, scanners per symbol etc.)
    # इस block को तुम्हारे final working logic से रखो; मैंने ऊपर UI part ही बदला है।
    pass  # ← इसे अपनी actual main scan logic से replace करो (जो अभी 127 वाले code में है)


# ==============================
# RESULT TABLE + TV LINK (जैसा तुमने पहले बनाया)
# ==============================
# df_res भरने के बाद:
# df_res["TV_Link"] = df_res["Symbol"].apply(
#     lambda s: f"[TV]({make_tradingview_link(s)})" if s else ""
# )
# df_res = df_res[SAFE_COLS]
# df_res = df_res.replace([np.inf, -np.inf], "").fillna("")
# st.markdown(df_res.to_markdown(index=False), unsafe_allow_html=True)

# ==============================
# RSI DONUT (Plotly polished)
# ==============================
if scanner == "RSI Market Pulse" and not df_res.empty:
    zone_counts = df_res["Zone"].value_counts().reset_index()
    zone_counts.columns = ["Zone", "Count"]

    fig = px.pie(
        zone_counts,
        names="Zone",
        values="Count",
        hole=0.65,
        color="Zone",
        color_discrete_map={
            "RSI > 60": "#22c55e",
            "RSI 40–60": "#eab308",
            "RSI < 40": "#ef4444",
        },
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        pull=[0.03] * len(zone_counts),
    )
    fig.update_layout(
        title=dict(
            text="RSI Market Pulse Distribution",
            font=dict(size=18, color="#e5e7eb", family="Inter"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(color="#e5e7eb"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=40, l=0, r=0),
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================
# SINGLE STOCK MATRIX (final section)
# ==============================
st.markdown("---")
st.markdown(
    """
    <div class="glass-card" style="margin-bottom:0.4rem;">
        <div style="font-size:15px; font-weight:600; color:#e5e7eb;">Scanner Matrix for Selected Stock</div>
        <div style="font-size:12px; color:#9ca3af;">
            Quick view of where the chosen symbol is firing across all scanners.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if selected_symbol != "NA":
    data_single_tf = load_data(TIMEFRAMES[tf])
    if selected_symbol in data_single_tf:
        df_sym = trim_df_to_date(data_single_tf[selected_symbol], analysis_date)
        if df_sym is not None:
            data_all_tfs = {
                tf: data_single_tf,
                "1 Hour": load_data(TIMEFRAMES["1 Hour"]),
                "Daily": load_data(TIMEFRAMES["Daily"]),
                "Weekly": load_data(TIMEFRAMES["Weekly"]),
                "Monthly": load_data(TIMEFRAMES["Monthly"]),
            }

            results_dict = run_all_scanners_for_symbol(
                selected_symbol,
                df_sym,
                tf,
                analysis_date,
                data_all_tfs,
            )

            mat_df = pd.DataFrame(
                {
                    "Scanner": list(results_dict.keys()),
                    "Result": ["Yes" if v else "No" for v in results_dict.values()],
                }
            )
            st.dataframe(mat_df, use_container_width=True, hide_index=True)
        else:
            st.info("Not enough data for this symbol at selected date.")
    else:
        st.info("Symbol data not found for this timeframe.")
