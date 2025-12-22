import os
import glob
import numpy as np
import pandas as pd
import talib
import streamlit as st
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

            if last_dt is None or dt > last_dt:
                last_dt = dt

        except Exception:
            continue

    return last_dt



# =========================================================
# CONFIG
# =========================================================






FOLDERS = {
    "D":   "stock_data_D",
    "W":   "stock_data_W",
    "M":   "stock_data_M",
    "15m": "stock_data_15",
    "1h":  "stock_data_1H",
}

FILTER_OPTIONS = [
    "MACD uptick", "MACD downtick", "MACD > 0", "MACD < 0",
    "MACD PCO", "MACD NCO",
    "51326_PCO", "51326_NCO",
    "UBBC", "LBBC",
    "Price > Med", "Price < Med",
    "Ungli",
    "DI Bull", "DI Bear",
    "Stoch PCO", "Stoch NCO",
]

FILTER_COLUMN_MAP = {
    "MACD uptick": "macd_uptick",
    "MACD downtick": "macd_downtick",
    "MACD > 0": "macd_pos",
    "MACD < 0": "macd_neg",
    "MACD PCO": "macd_pco",
    "MACD NCO": "macd_nco",
    "51326_PCO": "ema_51326_pco",
    "51326_NCO": "ema_51326_nco",
    "UBBC": "ubb_c",
    "LBBC": "lbbc_c",
    "Price > Med": "price_gt_med",
    "Price < Med": "price_lt_med",
    "Ungli": "ungli",
    "DI Bull": "di_bull",
    "DI Bear": "di_bear",
    "Stoch PCO": "stoch_pco",
    "Stoch NCO": "stoch_nco",
}

UI_COLORS = {
    "orange": "#F39C12",
    "blue": "#3498DB",
    "green": "#2ECC71",
    "red": "#E74C3C",
    "black": "#101010",
}

# =========================================================
# INDICATORS (TA-Lib)
# =========================================================

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    df["ema5"] = talib.EMA(close, timeperiod=5)
    df["ema13"] = talib.EMA(close, timeperiod=13)
    df["ema20"] = talib.EMA(close, timeperiod=20)
    df["ema26"] = talib.EMA(close, timeperiod=26)
    df["ema50"] = talib.EMA(close, timeperiod=50)

    df["rsi"] = talib.RSI(close, timeperiod=14)

    macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df["macd"] = macd
    df["signal"] = signal
    df["macd_hist"] = hist

    bb_up, bb_mid, bb_low = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    df["bb_up"] = bb_up
    df["bb_mid"] = bb_mid
    df["bb_low"] = bb_low

    k, d = talib.STOCH(high, low, close,
                       fastk_period=14,
                       slowk_period=3, slowk_matype=0,
                       slowd_period=3, slowd_matype=0)
    df["k"] = k
    df["d"] = d

    df["adx"] = talib.ADX(high, low, close, timeperiod=14)
    df["plus_di"] = talib.PLUS_DI(high, low, close, timeperiod=14)
    df["minus_di"] = talib.MINUS_DI(high, low, close, timeperiod=14)

    return df.dropna()


# =========================================================
# FLAGS / ABBREVIATIONS
# =========================================================

def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    # MACD uptick / downtick
    df["macd_uptick"] = df["macd"] > df["macd"].shift(1)
    df["macd_downtick"] = df["macd"] < df["macd"].shift(1)

    # MACD > 0 / < 0
    df["macd_pos"] = df["macd"] > 0
    df["macd_neg"] = df["macd"] < 0

    # MACD PCO / NCO
    df["macd_pco"] = df["macd"] > df["signal"]
    df["macd_nco"] = df["macd"] < df["signal"]

    # EMA 5 > 13 > 26 and 5 < 13 < 26
    df["ema_51326_pco"] = (df["ema5"] > df["ema13"]) & (df["ema13"] > df["ema26"])
    df["ema_51326_nco"] = (df["ema5"] < df["ema13"]) & (df["ema13"] < df["ema26"])

    # Bollinger changes
    df["ubb_c"] = df["bb_up"] > df["bb_up"].shift(1)
    df["lbbc_c"] = df["bb_low"] < df["bb_low"].shift(1)

    # Price vs median
    df["price_gt_med"] = df["close"] > df["bb_mid"]
    df["price_lt_med"] = df["close"] < df["bb_mid"]

    # ADX Ungli
    df["ungli"] = (
        (df["adx"] > 14) &
        (df["adx"] > df["adx"].shift(1)) &
        (df["adx"].shift(1) < df["adx"].shift(2))
    )

    # DI Bull / Bear
    df["di_bull"] = df["plus_di"] > df["minus_di"]
    df["di_bear"] = df["plus_di"] < df["minus_di"]

    # Stoch PCO / NCO
    df["stoch_pco"] = df["k"] > df["d"]
    df["stoch_nco"] = df["k"] < df["d"]

    return df


# =========================================================
# LOAD LATEST ROW PER SYMBOL FOR ONE TIMEFRAME
# =========================================================

def load_latest_from_folder(folder_path: str) -> pd.DataFrame:
    if not os.path.isdir(folder_path):
        return pd.DataFrame()

    rows = []
    for fname in os.listdir(folder_path):
        if not fname.endswith(".parquet"):
            continue
        fpath = os.path.join(folder_path, fname)
        try:
            df = pd.read_parquet(fpath)
            if df.empty:
                continue
            df = df.sort_index()
            df = compute_indicators(df)
            df = add_flags(df)
            rows.append(df.iloc[-1])
        except Exception as e:
            # Optional: log error if needed
            # st.write(f"Error reading {fpath}: {e}")
            continue

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result.set_index("symbol", inplace=True)
    return result


# =========================================================
# CORE SCAN FUNCTION (SAME LOGIC AS TKINTER VERSION)
# =========================================================

def run_scan(
    tf1: str,
    tf2: str,
    tf1_filters: list,
    tf2_filters: list,
    rsi1_cond: str,
    rsi1_val,
    rsi2_cond: str,
    rsi2_val,
) -> pd.DataFrame:

    df1 = load_latest_from_folder(FOLDERS[tf1])
    df2 = load_latest_from_folder(FOLDERS[tf2])

    if df1.empty or df2.empty:
        return pd.DataFrame()

    df1 = df1.add_suffix(f"_{tf1}")
    df2 = df2.add_suffix(f"_{tf2}")
    merged = df1.join(df2, how="inner")

    # Apply TF1 filters
    for cond in tf1_filters:
        if cond and cond != "None":
            base = FILTER_COLUMN_MAP[cond]
            col = f"{base}_{tf1}"
            if col in merged.columns:
                merged = merged[merged[col] == True]

    # Apply TF2 filters
    for cond in tf2_filters:
        if cond and cond != "None":
            base = FILTER_COLUMN_MAP[cond]
            col = f"{base}_{tf2}"
            if col in merged.columns:
                merged = merged[merged[col] == True]

    # RSI TF1
    rsi_col_1 = f"rsi_{tf1}"
    if rsi1_cond != "None" and rsi_col_1 in merged.columns and rsi1_val is not None:
        if rsi1_cond == "RSI >=":
            merged = merged[merged[rsi_col_1] >= rsi1_val]
        elif rsi1_cond == "RSI <=":
            merged = merged[merged[rsi_col_1] <= rsi1_val]

    # RSI TF2
    rsi_col_2 = f"rsi_{tf2}"
    if rsi2_cond != "None" and rsi_col_2 in merged.columns and rsi2_val is not None:
        if rsi2_cond == "RSI >=":
            merged = merged[merged[rsi_col_2] >= rsi2_val]
        elif rsi2_cond == "RSI <=":
            merged = merged[merged[rsi_col_2] <= rsi2_val]

    return merged


# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(
    page_title="Multi TF Multi-Condition Scanner – By GS",
    page_icon="💹",
    layout="wide",
)

# =====================================================
# LAST CANDLE DATES PER TIMEFRAME (FROM SOURCE DATA)
# =====================================================
last_daily = get_last_candle_by_tf(FOLDERS["D"])
last_15m   = get_last_candle_by_tf(FOLDERS["15m"])
last_1h    = get_last_candle_by_tf(FOLDERS["1h"])



# ---- Top title area ----
st.markdown(
    f"""
    <h1 style="text-align:center; color:{UI_COLORS['black']};">
        Multi TF Multi-Condition Scanner – By GS
    </h1>
    """,
    unsafe_allow_html=True,
)
# =====================================================
# DISPLAY LAST CANDLE INFO
# =====================================================
st.markdown(
    f"""
    <div style="text-align:center; font-size:15px; color:{UI_COLORS['green']};
                margin-bottom:10px;">

        🗓 <b>Daily</b>: {last_daily.date() if last_daily else 'NA'}
        &nbsp;&nbsp; | &nbsp;&nbsp;
        ⏱ <b>15 Min</b>: {last_15m.strftime('%d %b %Y %H:%M') if last_15m else 'NA'}
        &nbsp;&nbsp; | &nbsp;&nbsp;
        ⏰ <b>1 Hour</b>: {last_1h.strftime('%d %b %Y %H:%M') if last_1h else 'NA'}

    </div>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# =====================================================
# TOP DATA REFRESH CONTROL (IST - SAFE)
# =====================================================
ist_now = pd.Timestamp.now(tz="Asia/Kolkata")

col1, col2 = st.columns([1, 4])

with col1:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.success("Fresh data loaded from GitHub.")
        st.rerun()

with col2:
    st.caption(
        f"🕒 Last refresh (IST): {ist_now.strftime('%d %b %Y, %I:%M:%S %p')}"
    )


st.markdown(
    f"""
    <div style="text-align:right; color:{UI_COLORS['blue']};
                font-size:18px; font-weight:bold; margin-bottom:10px;">
        Designed by GS Yadav
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ---- Sidebar controls ----
with st.sidebar:
    st.markdown(
        f"<h3 style='color:{UI_COLORS['orange']}'>Timeframe & Filters</h3>",
        unsafe_allow_html=True,
    )

    tf1 = st.selectbox("TimeFrame 1 (TF1)", list(FOLDERS.keys()), index=0)
    tf2 = st.selectbox("TimeFrame 2 (TF2)", list(FOLDERS.keys()), index=1)

    st.markdown("### RSI Filters")
    rsi1_cond = st.selectbox("RSI TF1", ["None", "RSI >=", "RSI <="], index=0)
    rsi1_val = None
    if rsi1_cond != "None":
        rsi1_val = st.number_input("RSI TF1 Value", min_value=0.0, max_value=100.0, value=60.0, step=1.0)

    rsi2_cond = st.selectbox("RSI TF2", ["None", "RSI >=", "RSI <="], index=0)
    rsi2_val = None
    if rsi2_cond != "None":
        rsi2_val = st.number_input("RSI TF2 Value", min_value=0.0, max_value=100.0, value=60.0, step=1.0)

    st.markdown("### TF1 Conditions (5)")
    tf1_filters = []
    cond_options = ["None"] + FILTER_OPTIONS
    for i in range(5):
        v = st.selectbox(f"TF1 Cond {i+1}", cond_options, index=0, key=f"tf1_cond_{i}")
        tf1_filters.append(v)

    st.markdown("### TF2 Conditions (5)")
    tf2_filters = []
    for i in range(5):
        v = st.selectbox(f"TF2 Cond {i+1}", cond_options, index=0, key=f"tf2_cond_{i}")
        tf2_filters.append(v)

    run_btn = st.button("🔍 Run Scan")
    reset_btn = st.button("♻ Reset Filters")

# ---- Reset logic (just reload page in Streamlit world) ----
if reset_btn:
    st.rerun()

# ---- Main area ----
result_placeholder = st.empty()
count_placeholder = st.empty()
download_placeholder = st.empty()

if run_btn:
    with st.spinner("Running scan on both timeframes..."):
        df_res = run_scan(
            tf1=tf1,
            tf2=tf2,
            tf1_filters=tf1_filters,
            tf2_filters=tf2_filters,
            rsi1_cond=rsi1_cond,
            rsi1_val=rsi1_val,
            rsi2_cond=rsi2_cond,
            rsi2_val=rsi2_val,
        )

    if df_res.empty:
        result_placeholder.warning("No stocks match the selected conditions.")
    else:
        df_show = df_res.reset_index()

        # Show only some main columns for clarity
        show_cols = [
            "symbol",
            f"close_{tf1}", f"rsi_{tf1}", f"macd_{tf1}",
            f"close_{tf2}", f"rsi_{tf2}", f"macd_{tf2}",
        ]
        show_cols = [c for c in show_cols if c in df_show.columns]

        count_placeholder.markdown(
            f"<h4 style='color:{UI_COLORS['green']}'>Total Stocks: {len(df_show)}</h4>",
            unsafe_allow_html=True,
        )

        result_placeholder.dataframe(df_show[show_cols], use_container_width=True)

        # CSV download
        csv_data = df_show.to_csv(index=False).encode("utf-8")
        download_placeholder.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"GS_Scanner_{tf1}_{tf2}.csv",
            mime="text/csv",
        )
else:
    st.info("Configure filters in the sidebar and click **Run Scan**.")


st.markdown("""
---
**Designed by:-  
Gaurav Singh Yadav**   
🩷💛🩵💙🩶💜🤍🤎💖  Built With Love 🫶  
Energy | Commodity | Quant Intelligence 📶  
📱 +91-8003994518 〽️   
📧 yadav.gauravsingh@gmail.com ™️
""")















