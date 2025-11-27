import os
import glob
import numpy as np
import pandas as pd
import talib
import streamlit as st
import random
import smtplib
from email.mime.text import MIMEText


# ======================================================
# MUST BE FIRST UI COMMAND
# ======================================================
st.set_page_config(
    page_title="Multi TF Multi-Condition Scanner – By GS",
    layout="wide",
)


# ======================================================
# OTP AUTHENTICATION
# ======================================================
def send_otp(email):
    otp = str(random.randint(100000, 999999))
    st.session_state["otp"] = otp

    msg = MIMEText(f"Your OTP for GS Scanner login is: {otp}")
    msg["Subject"] = "OTP Login"
    msg["From"] = st.secrets.EMAIL_ID
    msg["To"] = email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(st.secrets.EMAIL_ID, st.secrets.EMAIL_PASS)
        server.send_message(msg)


def otp_login():

    if "verified" not in st.session_state:
        st.session_state.verified = False

    if not st.session_state.verified:
        st.title("🔐 GS Scanner Login")

        email = st.text_input("Enter Email Address")

        if st.button("Send OTP"):
            send_otp(email)
            st.success("OTP sent to your email ✔")

        otp = st.text_input("Enter OTP", type="password")

        if st.button("Verify OTP"):
            if otp == st.session_state.get("otp"):
                st.session_state.verified = True
                st.success("Login Successful 🎯")
                st.experimental_rerun()   # <<<<< REQUIRED
            else:
                st.error("Invalid OTP ❌")

        st.stop()


# ======================================================
# CALL OTP FIRST
# ======================================================
otp_login()


# ======================================================
# BELOW IS YOUR ACTUAL APP
# ======================================================

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
# INDICATORS
# =========================================================

def compute_indicators(df):
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    df["ema5"] = talib.EMA(close, timeperiod=5)
    df["ema13"] = talib.EMA(close, timeperiod=13)
    df["ema20"] = talib.EMA(close, timeperiod=20)
    df["ema26"] = talib.EMA(close, timeperiod=26)
    df["ema50"] = talib.EMA(close, timeperiod=50)

    df["rsi"] = talib.RSI(close, timeperiod=14)
    macd, signal, hist = talib.MACD(close)

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


def add_flags(df):
    df["macd_uptick"] = df["macd"] > df["macd"].shift(1)
    df["macd_downtick"] = df["macd"] < df["macd"].shift(1)
    df["macd_pos"] = df["macd"] > 0
    df["macd_neg"] = df["macd"] < 0
    df["macd_pco"] = df["macd"] > df["signal"]
    df["macd_nco"] = df["macd"] < df["signal"]

    df["ema_51326_pco"] = (df["ema5"] > df["ema13"]) & (df["ema13"] > df["ema26"])
    df["ema_51326_nco"] = (df["ema5"] < df["ema13"]) & (df["ema13"] < df["ema26"])

    df["ubb_c"] = df["bb_up"] > df["bb_up"].shift(1)
    df["lbbc_c"] = df["bb_low"] < df["bb_low"].shift(1)

    df["price_gt_med"] = df["close"] > df["bb_mid"]
    df["price_lt_med"] = df["close"] < df["bb_mid"]

    df["ungli"] = (
        (df["adx"] > 14)
        & (df["adx"] > df["adx"].shift(1))
        & (df["adx"].shift(1) < df["adx"].shift(2))
    )

    df["di_bull"] = df["plus_di"] > df["minus_di"]
    df["di_bear"] = df["plus_di"] < df["minus_di"]

    df["stoch_pco"] = df["k"] > df["d"]
    df["stoch_nco"] = df["k"] < df["d"]

    return df


# =========================================================
# LOAD FOLDER + MERGE + FILTER
# =========================================================

def load_latest_from_folder(folder):
    rows = []
    for f in os.listdir(folder):
        if f.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(folder, f))
            df = compute_indicators(df)
            df = add_flags(df)
            rows.append(df.iloc[-1])

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result.set_index("symbol", inplace=True)
    return result


# ********** Your existing UI + Scanner continues below **********
