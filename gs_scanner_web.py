import os
import glob
import numpy as np
import pandas as pd
import talib
import streamlit as st
import random
import smtplib
from email.mime.text import MIMEText

# ================================================
# OTP AUTH
# ================================================
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
    st.title("🔐 GS Scanner Login - OTP Required")

    if "verified" not in st.session_state:
        st.session_state.verified = False

    if not st.session_state.verified:
        email = st.text_input("Enter Email Address")

        if st.button("Send OTP"):
            send_otp(email)
            st.success("OTP sent to your email address")

        otp = st.text_input("Enter OTP")

        if st.button("Verify"):
            if otp == st.session_state.get("otp"):
                st.session_state.verified = True
                st.success("Login Successful ✔")
            else:
                st.error("Invalid OTP ❌")

        st.stop()

# call OTP screen before rendering app
otp_login()   # <---- VERY IMPORTANT


# =========================================================
# BELOW IS YOUR ORIGINAL CODE (UNTOUCHED)
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

# ============= rest of your code continues exactly same =============
# (no change needed)
