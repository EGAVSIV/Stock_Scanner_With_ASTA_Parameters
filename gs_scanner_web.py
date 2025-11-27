import os
import numpy as np
import pandas as pd
import talib
import streamlit as st
import random
import smtplib
from email.mime.text import MIMEText


# =========================================================
# STREAMLIT CONFIG – MUST BE FIRST
# =========================================================
st.set_page_config(
    page_title="Multi TF Multi Condition Scanner – By GS",
    layout="wide",
)


# =========================================================
# OTP AUTH LOGIC
# =========================================================
import streamlit as st
import random
import smtplib
from email.mime.text import MIMEText

def send_otp(email):
    otp = str(random.randint(100000, 999999))
    st.session_state["otp"] = otp

    msg = MIMEText(f"Your OTP for GS Scanner login: {otp}")
    msg["Subject"] = "GS Scanner OTP Login"
    msg["From"] = st.secrets.EMAIL_ID
    msg["To"] = email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(st.secrets.EMAIL_ID, st.secrets.EMAIL_PASS)
        server.send_message(msg)


def otp_login():
    if "verified" not in st.session_state:
        st.session_state.verified = False

    if st.session_state.verified:
        return  # <-- IMPORTANT (this avoids rerun crash)

    st.title("🔐 GS Scanner Login")

    email = st.text_input("Enter Email Address")

    if st.button("Send OTP"):
        send_otp(email)
        st.success("OTP sent to your email")

    otp = st.text_input("Enter OTP")

    if st.button("Verify"):
        if otp == st.session_state.get("otp"):
            st.session_state.verified = True
            st.success("Login Successful ✔")

            st.stop()  # exit screen cleanly
        else:
            st.error("Invalid OTP ❌")

    st.stop()  # keep showing OTP page



# =========================================================
# AUTH MUST RUN FIRST
# =========================================================
otp_login()


# =========================================================
# SCANNER CONFIG
# =========================================================

FOLDERS = {
    "D": "stock_data_D",
    "W": "stock_data_W",
    "M": "stock_data_M",
    "15m": "stock_data_15",
    "1h": "stock_data_1H",
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


# =========================================================
# INDICATOR CALCULATION
# =========================================================
def compute_indicators(df):
    close = df.close.astype(float)
    high = df.high.astype(float)
    low = df.low.astype(float)

    df["ema5"] = talib.EMA(close, 5)
    df["ema13"] = talib.EMA(close, 13)
    df["ema20"] = talib.EMA(close, 20)
    df["ema26"] = talib.EMA(close, 26)
    df["ema50"] = talib.EMA(close, 50)

    df["rsi"] = talib.RSI(close, 14)

    macd, signal, _ = talib.MACD(close)
    df["macd"] = macd
    df["signal"] = signal

    bb_up, bb_mid, bb_low = talib.BBANDS(close)
    df["bb_up"] = bb_up
    df["bb_mid"] = bb_mid
    df["bb_low"] = bb_low

    k, d = talib.STOCH(high, low, close)
    df["k"] = k
    df["d"] = d

    df["adx"] = talib.ADX(high, low, close)
    df["plus_di"] = talib.PLUS_DI(high, low, close)
    df["minus_di"] = talib.MINUS_DI(high, low, close)

    return df.dropna()


# =========================================================
# FLAGS
# =========================================================
def add_flags(df):
    df["macd_uptick"] = df.macd > df.macd.shift(1)
    df["macd_downtick"] = df.macd < df.macd.shift(1)
    df["macd_pos"] = df.macd > 0
    df["macd_neg"] = df.macd < 0
    df["macd_pco"] = df.macd > df.signal
    df["macd_nco"] = df.macd < df.signal
    df["ema_51326_pco"] = (df.ema5 > df.ema13) & (df.ema13 > df.ema26)
    df["ema_51326_nco"] = (df.ema5 < df.ema13) & (df.ema13 < df.ema26)
    df["ubb_c"] = df.bb_up > df.bb_up.shift(1)
    df["lbbc_c"] = df.bb_low < df.bb_low.shift(1)
    df["price_gt_med"] = df.close > df.bb_mid
    df["price_lt_med"] = df.close < df.bb_mid
    df["ungli"] = (df.adx > 14) & (df.adx > df.adx.shift(1)) & (df.adx.shift(1) < df.adx.shift(2))
    df["di_bull"] = df.plus_di > df.minus_di
    df["di_bear"] = df.plus_di < df.minus_di
    df["stoch_pco"] = df.k > df.d
    df["stoch_nco"] = df.k < df.d
    return df


# =========================================================
# LOAD + FILTER
# =========================================================
def load_latest_from_folder(folder):
    rows = []
    for file in os.listdir(folder):
        if file.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(folder, file))
            df = compute_indicators(df)
            df = add_flags(df)
            rows.append(df.iloc[-1])

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result.set_index("symbol", inplace=True)
    return result


# =========================================================
# UI – MAIN SCANNER
# =========================================================
st.title("📊 Multi Time-Frame Multi-Condition Scanner – GS")

if st.button("Logout"):
    st.session_state.verified = False
    st.experimental_rerun()


tf1 = st.selectbox("TimeFrame 1", list(FOLDERS.keys()))
tf2 = st.selectbox("TimeFrame 2", list(FOLDERS.keys()))

tf1_filters = [st.selectbox(f"TF1 Filter {i+1}", ["None"] + FILTER_OPTIONS) for i in range(5)]
tf2_filters = [st.selectbox(f"TF2 Filter {i+1}", ["None"] + FILTER_OPTIONS) for i in range(5)]

run = st.button("Run Scan")


def run_scan():
    df1 = load_latest_from_folder(FOLDERS[tf1])
    df2 = load_latest_from_folder(FOLDERS[tf2])

    df1 = df1.add_suffix(f"_{tf1}")
    df2 = df2.add_suffix(f"_{tf2}")
    df = df1.join(df2, how="inner")

    for cond in tf1_filters:
        if cond != "None":
            df = df[df[f"{FILTER_COLUMN_MAP[cond]}_{tf1}"]]

    for cond in tf2_filters:
        if cond != "None":
            df = df[df[f"{FILTER_COLUMN_MAP[cond]}_{tf2}"]]

    return df


if run:
    result = run_scan()
    st.write("### Filtered Stocks:", len(result))
    st.dataframe(result.reset_index())


