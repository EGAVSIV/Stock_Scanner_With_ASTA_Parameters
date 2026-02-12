import streamlit as st
import pandas as pd
import os
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="Market Dashboard",
    layout="wide",
    page_icon="Assets/sgy1.png"
)

# =====================================================
# AUTO REFRESH
# =====================================================
st_autorefresh(interval=5000, key="refresh")

# =====================================================
# PATHS
# =====================================================
BASE_DIR = "market_data"
BROADER_PATH = os.path.join(BASE_DIR, "broader_index", "D")
SECTOR_PATH  = os.path.join(BASE_DIR, "sector_index", "D")
FNO_PATH     = os.path.join(BASE_DIR, "fno", "D")
SECTOR_MAP_FILE = "FNOSECTOR.xlsx"

# =====================================================
# HELPERS
# =====================================================
def load_bucket(folder):
    rows = []
    for f in os.listdir(folder):
        if not f.endswith(".parquet"):
            continue
        symbol = f.replace(".parquet", "")
        try:
            df = pd.read_parquet(os.path.join(folder, f)).sort_index()
            if len(df) >= 2:
                p, c = df["close"].iloc[-2], df["close"].iloc[-1]
                change = round(((c - p) / p) * 100, 2) if p else 0
            else:
                change = 0
        except:
            change = 0
        rows.append({"Symbol": symbol, "Change": change})
    return pd.DataFrame(rows)

def assign_colors(df):
    max_pos = df[df["Change"] > 0]["Change"].max() or 1
    min_neg = df[df["Change"] < 0]["Change"].min() or -1

    colors = []
    for v in df["Change"]:
        if v >= 0:
            intensity = v / max_pos
            colors.append(f"rgba(0,180,0,{0.3 + 0.7*intensity})")
        else:
            intensity = abs(v / min_neg)
            colors.append(f"rgba(220,0,0,{0.3 + 0.7*intensity})")
    return colors

# =====================================================
# HEADER
# =====================================================
col_logo, col_text = st.columns([0.2, 0.8])
with col_logo:
    st.image("Assets/sgy1.png", width=180)

with col_text:
    st.markdown("""
    <div style="background:#141E30;color:white;
    padding:14px;border-radius:10px;
    font-size:22px;font-weight:700">
    📜 याद रखना कमजोर हम नहीं, हमारा वक्त है ⏩ मेहनत जारी रखो ⏩ हार तब होती है जब मान लिया जाए ⏩ खामोशी से मेहनत करो, शोर खुद बन जाएगा ⏩खुद पर भरोसा रखो, यही सबसे बड़ी ताकत है  ⏩  हमारी समस्या का समाधान सिर्फ हमारे पास है, दूसरों के पास तो सिर्फ सुझाव है⏩
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =====================================================
# LOAD DATA
# =====================================================
df_broader = load_bucket(BROADER_PATH)
df_sector  = load_bucket(SECTOR_PATH)
df_fno     = load_bucket(FNO_PATH)

top5 = df_fno.sort_values("Change", ascending=False).head(5)
bottom5 = df_fno.sort_values("Change").head(5)

# =====================================================
# GAINERS / LOSERS TICKER
# =====================================================
gainers = " | ".join([f"{r.Symbol} +{r.Change}%" for r in top5.itertuples()])
losers  = " | ".join([f"{r.Symbol} {r.Change}%" for r in bottom5.itertuples()])

st.markdown(f"""
<style>
.ticker {{
  white-space: nowrap;
  animation: scroll 10s linear infinite;
}}
@keyframes scroll {{
  0% {{ transform: translateX(100%); }}
  100% {{ transform: translateX(-100%); }}
}}
</style>

<div style="background:#eaffea;padding:8px;border-radius:8px">
<span class="ticker">🟢 FNO GAINERS → {gainers}</span>
</div>
<div style="background:#ffeaea;padding:8px;border-radius:8px;margin-top:6px">
<span class="ticker">🔴 FNO LOSERS → {losers}</span>
</div>
""", unsafe_allow_html=True)

# =====================================================
# DASHBOARD
# =====================================================
st.title("📊 Market Dashboard")

adv = (df_fno["Change"] > 0).sum()
dec = (df_fno["Change"] < 0).sum()
unch = (df_fno["Change"] == 0).sum()

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.subheader("📈 Broad Market Indices")
    df = df_broader.sort_values("Change")
    fig = px.bar(df, x="Change", y="Symbol", orientation="h")
    fig.update_traces(marker_color=assign_colors(df))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("🏭 Sector Performance")
    df = df_sector.sort_values("Change")
    fig = px.bar(df, x="Change", y="Symbol", orientation="h")
    fig.update_traces(marker_color=assign_colors(df))
    st.plotly_chart(fig, use_container_width=True)

with c3:
    st.subheader("⚖️ Advance / Decline (FNO)")

    fig = px.pie(
        names=["Advance", "Decline", "Unchanged"],
        values=[adv, dec, unch],
        hole=0.45,
        color=["Advance", "Decline", "Unchanged"],
        color_discrete_map={
            "Advance": "#00C853",     # GREEN
            "Decline": "#D50000",     # RED
            "Unchanged": "#9E9E9E"    # GREY
        }
    )

    fig.update_traces(
        textinfo="percent+label",
        marker=dict(line=dict(color="white", width=2))
    )

    st.plotly_chart(fig, use_container_width=True)


with c4:
    st.subheader("🔥 FNO Top 5 & Bottom 5")
    df = pd.concat([top5, bottom5]).sort_values("Change")
    fig = px.bar(df, x="Change", y="Symbol", orientation="h")
    fig.update_traces(marker_color=assign_colors(df))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# =====================================================
# SECTOR DRILLDOWN
# =====================================================
sector_map = pd.read_excel(SECTOR_MAP_FILE)
sector_map.columns = sector_map.columns.str.lower().str.strip()
sector_map = sector_map.rename(
    columns={
        [c for c in sector_map.columns if "stock" in c][0]: "Stock",
        [c for c in sector_map.columns if "sector" in c][0]: "Sector"
    }
)

def sector_block(sector):
    stocks = sector_map[sector_map["Sector"] == sector]["Stock"]
    df = df_fno[df_fno["Symbol"].isin(stocks)].sort_values("Change")
    if df.empty:
        st.info("No stocks")
        return
    fig = px.bar(df, x="Change", y="Symbol", orientation="h")
    fig.update_traces(marker_color=assign_colors(df))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("🏭 Sector-wise Drilldown (Auto)")

df_sec_sorted = df_sector.sort_values("Change", ascending=False)
auto_sectors = list(df_sec_sorted.head(2)["Symbol"]) + list(df_sec_sorted.tail(2)["Symbol"])

cols = st.columns(4)
for col, sec in zip(cols, auto_sectors):
    with col:
        tag = "🟢 TOP" if sec in auto_sectors[:2] else "🔴 BOTTOM"
        st.markdown(f"### {tag} : {sec}")
        sector_block(sec)

st.markdown("---")
st.subheader("🔍 Manual Sector View")
sel_sector = st.selectbox("Select Sector", sorted(df_sector["Symbol"].tolist()))
sector_block(sel_sector)
