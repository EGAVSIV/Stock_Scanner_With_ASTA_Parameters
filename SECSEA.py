import streamlit as st
import pandas as pd
import os
import plotly.express as px
from datetime import datetime
import numpy as np

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Sector Seasonality Intelligence",
    layout="wide",
    page_icon="📆"
)

st.title("📆 Sector Seasonality Intelligence Engine")

REPO_NAME = "Stock_Scanner_With_ASTA_Parameters"
SECTOR_PATH = os.path.join(REPO_NAME, "market_data", "sector_index", "D")

if not os.path.exists(SECTOR_PATH):
    st.error(f"❌ Sector folder not found: {SECTOR_PATH}")
    st.stop()

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def get_nearest_available_date(df, target_date):

    if target_date in df.index:
        return target_date

    future_dates = df[df.index > target_date].index
    if len(future_dates) > 0:
        return future_dates[0]

    return min(df.index, key=lambda x: abs(x - target_date))


def calculate_period_return(file_path, start_md, end_md):

    df = pd.read_parquet(file_path)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if "close" not in df.columns:
        return {}

    years = sorted(df.index.year.unique())
    results = {}

    for year in years:
        try:
            start_date = pd.Timestamp(year=year, month=start_md.month, day=start_md.day)
            end_date = pd.Timestamp(year=year, month=end_md.month, day=end_md.day)

            if start_date > df.index.max():
                continue

            start_actual = get_nearest_available_date(df, start_date)
            end_actual = get_nearest_available_date(df, end_date)

            start_close = df.loc[start_actual]["close"]
            end_close = df.loc[end_actual]["close"]

            if pd.isna(start_close) or pd.isna(end_close):
                continue

            ret = round(((end_close - start_close) / start_close) * 100, 2)

            results[year] = ret

        except Exception:
            continue

    return results


def compute_strength_model(df):

    summary = pd.DataFrame(index=df.index)

    summary["Average Return"] = df.mean(axis=1)
    summary["Win Rate %"] = (df.gt(0).sum(axis=1) / df.count(axis=1)) * 100
    summary["Volatility (Std Dev)"] = df.std(axis=1)

    # Normalization
    avg_norm = (summary["Average Return"] - summary["Average Return"].min()) / \
               (summary["Average Return"].max() - summary["Average Return"].min())

    win_norm = summary["Win Rate %"] / 100

    vol_norm = 1 - (
        (summary["Volatility (Std Dev)"] - summary["Volatility (Std Dev)"].min()) /
        (summary["Volatility (Std Dev)"].max() - summary["Volatility (Std Dev)"].min())
    )

    summary["Strength Score"] = (
        avg_norm * 0.4 +
        win_norm * 0.3 +
        vol_norm * 0.3
    ) * 100

    summary = summary.sort_values("Strength Score", ascending=False)

    return summary


# =====================================================
# USER INPUT
# =====================================================

col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input("Select Start Date (Month & Day)")

with col2:
    end_date = st.date_input("Select End Date (Month & Day)")

# =====================================================
# MAIN LOGIC
# =====================================================

if start_date and end_date:

    if start_date >= end_date:
        st.warning("Start date must be less than End date")
        st.stop()

    seasonal_data = {}

    for file in os.listdir(SECTOR_PATH):
        if not file.endswith(".parquet"):
            continue

        sector = file.replace(".parquet", "")
        path = os.path.join(SECTOR_PATH, file)

        returns = calculate_period_return(path, start_date, end_date)

        if returns:
            seasonal_data[sector] = returns

    if not seasonal_data:
        st.warning("No seasonal data found.")
        st.stop()

    df_seasonal = pd.DataFrame(seasonal_data).T
    df_seasonal = df_seasonal[sorted(df_seasonal.columns, reverse=True)]
    df_seasonal = df_seasonal.sort_index()

    # =====================================================
    # TABLE VIEW
    # =====================================================

    st.markdown("---")
    st.subheader("📋 Seasonal Return Table")

    def style_cells(val):
        if pd.isna(val):
            return ""

        if val > 10:
            return "background-color:#004d00;color:white"
        elif val > 5:
            return "background-color:#006600;color:white"
        elif val > 0:
            return "background-color:#00b300;color:black"
        elif val < -10:
            return "background-color:#660000;color:white"
        elif val < -5:
            return "background-color:#990000;color:white"
        elif val < 0:
            return "background-color:#ff4d4d;color:black"
        return ""

    st.dataframe(
        df_seasonal.style.applymap(style_cells),
        use_container_width=True,
        height=500
    )

    # =====================================================
    # HEATMAP
    # =====================================================

    st.markdown("---")
    st.subheader("📊 Heatmap View")

    fig = px.imshow(
        df_seasonal,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdYlGn"
    )

    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # STRENGTH MODEL
    # =====================================================

    st.markdown("---")
    st.subheader("🚀 Sector Strength Model")

    df_strength = compute_strength_model(df_seasonal)

    st.dataframe(df_strength.round(2), use_container_width=True)

    fig_strength = px.bar(
        df_strength,
        x="Strength Score",
        y=df_strength.index,
        orientation="h",
        color="Strength Score",
        color_continuous_scale="RdYlGn"
    )

    fig_strength.update_layout(height=600)
    st.plotly_chart(fig_strength, use_container_width=True)

    # =====================================================
    # EXPORT TO EXCEL
    # =====================================================

    st.markdown("---")
    st.subheader("📊 Export Data")

    excel_file = "Sector_Seasonality_Report.xlsx"

    with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
        df_seasonal.to_excel(writer, sheet_name="Seasonal Returns")
        df_strength.to_excel(writer, sheet_name="Strength Model")

    with open(excel_file, "rb") as f:
        st.download_button(
            label="📥 Download Excel Report",
            data=f,
            file_name=excel_file,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
