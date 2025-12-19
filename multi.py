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
# ======== SCANNERS (UNCHANGED) ====================
# ==================================================
# 🔴 ALL YOUR FUNCTIONS HERE — UNTOUCHED
# (rsi_market_pulse … bearish_gsas)
# 👉 EXACTLY AS YOU PASTED
# ==================================================

# (Functions omitted here ONLY to save chat length —
# they must remain EXACTLY as in your pasted code)

# ==================================================
# SIDEBAR
# ==================================================
tf = st.sidebar.selectbox("Timeframe", list(TIMEFRAMES.keys()))

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

    # ✅ SAFE DEFAULTS (CRITICAL FIX)
    data_htf = {}
    data_w = {}
    data_m = {}

    # ✅ GSAS HTF LOGIC (FIXED)
    if scanner in ["Bullish GSAS", "Bearish GSAS"]:
        htf_map = {
            "15 Min": "1 Hour",
            "1 Hour": "Daily",
            "Daily": "Weekly",
            "Weekly": "Monthly",   # added as requested
        }

        if tf not in htf_map:
            st.warning("GSAS not supported for this timeframe")
            st.stop()

        data_htf = load_data(TIMEFRAMES[htf_map[tf]])

    # ✅ RSI WM PRELOAD
    if scanner == "RSI WM 60–40":
        data_w = load_data(TIMEFRAMES["Weekly"])
        data_m = load_data(TIMEFRAMES["Monthly"])

    for sym, df in data.items():

        # 🔴 YOUR ORIGINAL SCANNER LOGIC BELOW — UNCHANGED

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
            if sym in data_htf:
                sig = bullish_gsas(df, data_htf[sym])
                if sig:
                    results.append({"Symbol": sym, "Signal": sig})

        elif scanner == "Bearish GSAS":
            if sym in data_htf:
                sig = bearish_gsas(df, data_htf[sym])
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
