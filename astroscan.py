import time
import math
import datetime
from pathlib import Path

import pandas as pd
import swisseph as swe
import pytz
import matplotlib

# Use non-GUI backend for Streamlit
matplotlib.use("Agg")
from matplotlib.figure import Figure
import mplfinance as mpf
import streamlit as st

# ---------------------------------------------------------------------
# CONFIG / CONSTANTS
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Planetary Aspects & Stock Scanner — Web",
    layout="wide",
)

# --- swisseph setup (Lahiri) ---
swe.set_sid_mode(swe.SIDM_LAHIRI, 0, 0)

# --- Constants ---
NAK_DEG = 13 + 1 / 3

ZODIACS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

PLANETS = {
    "Sun": swe.SUN,
    "Moon": swe.MOON,
    "Mercury": swe.MERCURY,
    "Venus": swe.VENUS,
    "Mars": swe.MARS,
    "Jupiter": swe.JUPITER,
    "Saturn": swe.SATURN,
    "Rahu": swe.TRUE_NODE,  # Ketu = opposite
}

NAKSHATRAS = [
    ("Ashwini", "Ketu"), ("Bharani", "Venus"), ("Krittika", "Sun"), ("Rohini", "Moon"),
    ("Mrigashira", "Mars"), ("Ardra", "Rahu"), ("Punarvasu", "Jupiter"), ("Pushya", "Saturn"),
    ("Ashlesha", "Mercury"), ("Magha", "Ketu"), ("Purva Phalguni", "Venus"), ("Uttara Phalguni", "Sun"),
    ("Hasta", "Moon"), ("Chitra", "Mars"), ("Swati", "Rahu"), ("Vishakha", "Jupiter"),
    ("Anuradha", "Saturn"), ("Jyeshtha", "Mercury"), ("Mula", "Ketu"), ("Purva Ashadha", "Venus"),
    ("Uttara Ashadha", "Sun"), ("Shravana", "Moon"), ("Dhanishta", "Mars"), ("Shatabhisha", "Rahu"),
    ("Purva Bhadrapada", "Jupiter"), ("Uttara Bhadrapada", "Saturn"), ("Revati", "Mercury"),
]

ASPECTS = {
    "Opposition": {
        "Aries": "Libra", "Taurus": "Scorpio", "Gemini": "Sagittarius",
        "Cancer": "Capricorn", "Leo": "Aquarius", "Virgo": "Pisces",
        "Libra": "Aries", "Scorpio": "Taurus", "Sagittarius": "Gemini",
        "Capricorn": "Cancer", "Aquarius": "Leo", "Pisces": "Virgo",
    },
    "Conjunction": {z: z for z in ZODIACS},
    "Square": {
        "Aries": "Cancer", "Taurus": "Leo", "Gemini": "Virgo",
        "Cancer": "Libra", "Leo": "Scorpio", "Virgo": "Sagittarius",
        "Libra": "Capricorn", "Scorpio": "Aquarius", "Sagittarius": "Pisces",
        "Capricorn": "Aries", "Aquarius": "Taurus", "Pisces": "Gemini",
    },
    "Trine": {
        "Aries": "Leo", "Taurus": "Virgo", "Gemini": "Libra",
        "Cancer": "Scorpio", "Leo": "Sagittarius", "Virgo": "Capricorn",
        "Libra": "Aquarius", "Scorpio": "Pisces", "Sagittarius": "Aries",
        "Capricorn": "Taurus", "Aquarius": "Gemini", "Pisces": "Cancer",
    },
    "Sextile": {
        "Aries": "Gemini", "Taurus": "Cancer", "Gemini": "Leo",
        "Cancer": "Virgo", "Leo": "Libra", "Virgo": "Scorpio",
        "Libra": "Sagittarius", "Scorpio": "Capricorn", "Sagittarius": "Aquarius",
        "Capricorn": "Pisces", "Aquarius": "Aries", "Pisces": "Taurus",
    },
}

# *** IMPORTANT: default folder changed as per your request ***
DEFAULT_STOCK_FOLDER = "stock_data_D"


# ---------------------------------------------------------------------
# CORE LOGIC (unchanged from Tkinter version)
# ---------------------------------------------------------------------
def get_sidereal_lon_from_jd(jd, planet_code):
    res = swe.calc_ut(jd, planet_code)
    if isinstance(res, tuple) and isinstance(res[0], (list, tuple)):
        lon = res[0][0]
        speed = res[0][3]
    elif isinstance(res, (list, tuple)):
        lon = res[0]
        speed = res[3] if len(res) > 3 else 0.0
    else:
        lon = float(res[0])
        speed = float(res[3]) if len(res) > 3 else 0.0
    ayan = swe.get_ayanamsa_ut(jd)
    sid_lon = (lon - ayan) % 360
    return sid_lon, speed


def get_zodiac_name(sid_lon):
    sign_index = int(sid_lon // 30) % 12
    return ZODIACS[sign_index]


def format_planet_line(pname, sid_lon, speed):
    sign_index = int(sid_lon // 30) % 12
    sign_name = ZODIACS[sign_index]
    deg_in_sign = sid_lon % 30
    deg = int(deg_in_sign)
    minutes = int((deg_in_sign - deg) * 60)
    seconds = int((((deg_in_sign - deg) * 60) - minutes) * 60)
    full_deg = round(sid_lon, 3)
    nak_index = int(sid_lon / NAK_DEG) % len(NAKSHATRAS)
    nak_name, nak_lord = NAKSHATRAS[nak_index]
    nak_pada = int((sid_lon % NAK_DEG) / (NAK_DEG / 4)) + 1
    retro = "R" if (pname in ["Rahu", "Ketu"] or speed < 0) else "D"
    return (
        f"{pname:<6}: {deg:02d}° {sign_name[:3]} {minutes:02d}′ {seconds:02d}″ | "
        f"{full_deg:07.3f}° | {nak_name:12} Pada {nak_pada} | "
        f"Lord: {nak_lord:<7} | {retro}"
    )


def find_aspect_dates(
    planet1,
    planet2,
    aspect_name,
    years_back=10,
    years_forward=5,
    limit_past=20,
    limit_future=5,
):
    today = datetime.datetime.now()
    jd_today = swe.julday(
        today.year, today.month, today.day, today.hour + today.minute / 60.0
    )
    p1 = PLANETS[planet1]
    p2 = PLANETS[planet2]
    aspect_map = ASPECTS[aspect_name]
    results_past = []
    results_future = []

    start_offset = -365 * years_back
    end_offset = 365 * years_forward

    for offset in range(start_offset, end_offset + 1):
        jd = jd_today + offset
        lon1, _ = get_sidereal_lon_from_jd(jd, p1)
        lon2, _ = get_sidereal_lon_from_jd(jd, p2)
        z1 = get_zodiac_name(lon1)
        z2 = get_zodiac_name(lon2)
        if aspect_map.get(z1) == z2:
            y, m, d, hr = swe.revjul(jd)
            date_str = f"{d:02d}-{m:02d}-{y}"
            if offset < 0:
                results_past.append(date_str)
            else:
                results_future.append(date_str)

    def unique_first_past(entries, keep):
        out = []
        prev = None
        for e in entries:
            if prev is None or (
                datetime.datetime.strptime(e, "%d-%m-%Y")
                - datetime.datetime.strptime(prev, "%d-%m-%Y")
            ).days != 1:
                out.append(e)
            prev = e
        return out[-keep:][::-1]

    def unique_first_future(entries, keep):
        out = []
        prev = None
        for e in entries:
            if prev is None or (
                datetime.datetime.strptime(e, "%d-%m-%Y")
                - datetime.datetime.strptime(prev, "%d-%m-%Y")
            ).days != 1:
                out.append(e)
            prev = e
        return out[:keep]

    return unique_first_past(results_past, limit_past), unique_first_future(
        results_future, limit_future
    )


def load_parquet_for_symbol(filepath: Path):
    df = pd.read_parquet(filepath)
    if "datetime" in df.columns:
        df = df.set_index(pd.to_datetime(df["datetime"]))
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

    if "timeframe" in df.columns:
        df = df[df["timeframe"] == "D"]

    if "close" not in df.columns:
        raise ValueError("No 'close' column in parquet")

    return df.sort_index()


def analyze_symbol_for_aspect_dates(df: pd.DataFrame, aspect_dates):
    results = []
    for ds in aspect_dates:
        try:
            d = datetime.datetime.strptime(ds, "%d-%m-%Y").date()
        except Exception:
            continue
        mask = df.index.date == d
        if not mask.any():
            continue
        idx = df.index[mask][0]
        close_on_date = float(df.loc[idx, "close"])
        idx_pos = df.index.get_loc(idx)

        start_pos = idx_pos + 1
        end_pos = start_pos + 10
        window = df.iloc[start_pos:end_pos]
        if window.empty:
            continue
        max_next10 = float(window["close"].max())
        min_next10 = float(window["close"].min())
        pct_max = ((max_next10 - close_on_date) / close_on_date) * 100.0
        pct_min = ((min_next10 - close_on_date) / close_on_date) * 100.0

        results.append(
            {
                "aspect_date": ds,
                "close": close_on_date,
                "max10": max_next10,
                "min10": min_next10,
                "pct_max": pct_max,
                "pct_min": pct_min,
            }
        )
    return results


# ---------------------------------------------------------------------
# STATE + LOGGING HELPERS
# ---------------------------------------------------------------------
def init_session_state():
    defaults = {
        "stock_folder": str(Path(DEFAULT_STOCK_FOLDER).resolve()),
        "aspect_dates_past": [],
        "aspect_dates_future": [],
        "planet1": "Sun",
        "planet2": "Moon",
        "aspect": "Opposition",
        "scan_results": pd.DataFrame(),
        "logs": [],
        "live_tick_value": 26000.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def log(msg, level="INFO"):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{ts}] {level}: {msg}")


init_session_state()

# ---------------------------------------------------------------------
# SIDEBAR (global controls)
# ---------------------------------------------------------------------
st.sidebar.header("Settings")

st.sidebar.text("Stock Data Folder")
st.session_state.stock_folder = st.sidebar.text_input(
    "Folder containing parquet files",
    value=st.session_state.stock_folder,
    label_visibility="collapsed",
)

st.sidebar.caption("Default: ./stock_data_D")

st.sidebar.markdown("---")
if st.sidebar.button("Clear Logs"):
    st.session_state.logs = []

# ---------------------------------------------------------------------
# MAIN LAYOUT – TABS
# ---------------------------------------------------------------------
st.title("Planetary Aspects & Stock Scanner — Web Version")

tabs = st.tabs(["Aspects", "Stocks Scan", "Charts", "Live Panel", "Logs"])

# ---------------------------------------------------------------------
# TAB 1 – ASPECTS
# ---------------------------------------------------------------------
with tabs[0]:
    st.subheader("Find Planetary Aspect Dates")

    col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 1.5])

    with col1:
        st.session_state.planet1 = st.selectbox(
            "Planet 1", list(PLANETS.keys()), index=list(PLANETS.keys()).index(st.session_state.planet1)
        )

    with col2:
        st.session_state.planet2 = st.selectbox(
            "Planet 2", list(PLANETS.keys()), index=list(PLANETS.keys()).index(st.session_state.planet2)
        )

    with col3:
        st.session_state.aspect = st.selectbox(
            "Aspect", list(ASPECTS.keys()), index=list(ASPECTS.keys()).index(st.session_state.aspect)
        )

    with col4:
        years_back = st.number_input("Years back", min_value=1, max_value=50, value=10, step=1)
        years_forward = st.number_input("Years forward", min_value=1, max_value=50, value=5, step=1)

    if st.button("🔍 Find Aspect Dates"):
        with st.spinner("Computing aspect dates..."):
            p1 = st.session_state.planet1
            p2 = st.session_state.planet2
            asp = st.session_state.aspect
            log(f"Finding aspect dates: {p1} {asp} {p2}")
            past, future = find_aspect_dates(
                p1, p2, asp, years_back=years_back, years_forward=years_forward
            )
            st.session_state.aspect_dates_past = past
            st.session_state.aspect_dates_future = future
        st.success(f"Found {len(past)} past and {len(future)} future aspect dates.")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("**Past occurrences (most recent first)**")
        if st.session_state.aspect_dates_past:
            st.write(st.session_state.aspect_dates_past)
        else:
            st.info("No past dates computed yet.")

    with colB:
        st.markdown("**Future dates**")
        if st.session_state.aspect_dates_future:
            st.write(st.session_state.aspect_dates_future)
        else:
            st.info("No future dates computed yet.")

    st.markdown("---")
    st.markdown("### Apply Aspect Dates to Stock Scanner")

    if st.button("➡ Use these aspect dates in Stocks Scan"):
        if not st.session_state.aspect_dates_past:
            st.warning("No aspect dates. Please compute aspects first.")
        else:
            log("Aspect dates sent to stock scanner.")
            st.success("Aspect dates ready for Stocks Scan tab.")


# ---------------------------------------------------------------------
# TAB 2 – STOCKS SCAN
# ---------------------------------------------------------------------
with tabs[1]:
    st.subheader("Scan Stocks Around Aspect Dates")

    folder = Path(st.session_state.stock_folder)

    st.write(f"Using data folder: `{folder}`")

    if not folder.exists():
        st.error(f"Folder not found: {folder}")
    else:
        st.success("Folder found.")

    col_run, col_save, col_clear = st.columns(3)

    def run_scan():
        aspect_dates = st.session_state.aspect_dates_past
        if not aspect_dates:
            st.warning("No aspect dates available. Go to Aspects tab first.")
            return

        if not folder.exists():
            st.error(f"Folder not found: {folder}")
            return

        parquet_files = list(folder.glob("*.parquet"))
        if not parquet_files:
            st.error("No .parquet files found in folder.")
            return

        results = []
        total = len(parquet_files)

        with st.spinner("Scanning parquet files..."):
            for i, pf in enumerate(parquet_files, start=1):
                sym = pf.stem
                try:
                    df = load_parquet_for_symbol(pf)
                except Exception as e:
                    log(f"Failed to load {pf.name}: {e}", level="ERROR")
                    continue

                log(f"[{i}/{total}] Scanning {pf.name}")
                items = analyze_symbol_for_aspect_dates(df, aspect_dates)

                for it in items:
                    # same filter logic as original:
                    if (it["pct_max"] >= 10.0) or (it["pct_min"] <= -10.0):
                        aspect_type = (
                            f"{st.session_state.planet1} "
                            f"{st.session_state.aspect} "
                            f"{st.session_state.planet2}"
                        )

                        if it["pct_max"] >= 10.0:
                            move_category = "😆 >10% Gain"
                        elif it["pct_min"] <= -10.0:
                            move_category = "😩 >10% Fall"
                        else:
                            move_category = "Normal Move"

                        results.append(
                            {
                                "symbol": sym,
                                "aspect_date": it["aspect_date"],
                                "close": it["close"],
                                "max10": it["max10"],
                                "min10": it["min10"],
                                "pct_max": round(it["pct_max"], 2),
                                "pct_min": round(it["pct_min"], 2),
                                "Aspect": aspect_type,
                                "Move Category": move_category,
                            }
                        )

                time.sleep(0.01)

        df_res = pd.DataFrame(results)
        st.session_state.scan_results = df_res
        log(f"Scan complete. {len(df_res)} qualifying records found.")
        if df_res.empty:
            st.info("No qualifying records found (>=10% move).")
        else:
            st.success(f"Scan complete with {len(df_res)} records.")

    with col_run:
        if st.button("Run Scan (using aspect dates)"):
            run_scan()

    with col_save:
        if not st.session_state.scan_results.empty:
            csv_data = st.session_state.scan_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results as CSV",
                data=csv_data,
                file_name="scan_results.csv",
                mime="text/csv",
            )
        else:
            st.button("Download Results as CSV", disabled=True)

    with col_clear:
        if st.button("Clear Results"):
            st.session_state.scan_results = pd.DataFrame()
            log("Results cleared.")

    st.markdown("---")
    st.markdown("### Scan Results")

    if st.session_state.scan_results.empty:
        st.info("No results yet. Run a scan to see data here.")
    else:
        st.dataframe(st.session_state.scan_results, use_container_width=True)


# ---------------------------------------------------------------------
# TAB 3 – CHARTS (candlesticks around aspect date)
# ---------------------------------------------------------------------
with tabs[2]:
    st.subheader("Candlestick Charts Around Aspect Date")

    df_res = st.session_state.scan_results
    folder = Path(st.session_state.stock_folder)

    if df_res.empty:
        st.info("No scan results. Run a scan first.")
    else:
        symbols = sorted(df_res["symbol"].unique())
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            symbol = st.selectbox("Symbol", symbols)
        df_sym = df_res[df_res["symbol"] == symbol]
        with col_s2:
            aspect_date = st.selectbox("Aspect Date", df_sym["aspect_date"].unique())

        if st.button("Show Chart"):
            pf = folder / f"{symbol}.parquet"
            if not pf.exists():
                # fallback: any file containing symbol name
                candidates = list(folder.glob(f"*{symbol}*.parquet"))
                if candidates:
                    pf = candidates[0]
                else:
                    st.error(f"No parquet file found for {symbol}")
                    log(f"No parquet for {symbol}", level="ERROR")
                    st.stop()

            try:
                df = load_parquet_for_symbol(pf)
            except Exception as e:
                st.error(f"Error loading {pf}: {e}")
                log(f"Load error {pf}: {e}", level="ERROR")
                st.stop()

            d = datetime.datetime.strptime(aspect_date, "%d-%m-%Y").date()
            start = d - datetime.timedelta(days=30)
            end = d + datetime.timedelta(days=40)

            dfw = df[(df.index.date >= start) & (df.index.date <= end)]
            if dfw.empty:
                st.warning("No data around this aspect date to plot.")
                st.stop()

            required_cols = {"open", "high", "low", "close"}
            if not required_cols.issubset(dfw.columns):
                st.error("OHLC columns missing in parquet; cannot plot candles.")
                st.stop()

            df_candle = dfw[["open", "high", "low", "close"]].copy()
            fig = Figure(figsize=(10, 4))
            ax = fig.add_subplot(111)
            mpf.plot(
                df_candle,
                type="candle",
                ax=ax,
                style="charles",
                show_nontrading=True,
            )
            ax.set_title(f"{symbol} – Candles around {aspect_date}")
            ax.grid(True, alpha=0.3)

            # vertical line on aspect date
            try:
                dates = pd.Series(dfw.index)
                idx_near = dates[dates.dt.date == d]
                if not idx_near.empty:
                    ad_idx = idx_near.iloc[0]
                    y = dfw.loc[ad_idx, "close"]
                    ax.axvline(ad_idx, color="orange", linestyle="--", linewidth=1)
                    ax.scatter([ad_idx], [y], color="orange")
            except Exception:
                pass

            st.pyplot(fig)


# ---------------------------------------------------------------------
# TAB 4 – LIVE PANEL (simple simulation)
# ---------------------------------------------------------------------
with tabs[3]:
    st.subheader("Live Panel (Simulation)")

    st.write(
        "This panel simulates a simple live tick feed and sends lines to the Logs tab. "
        "In your real integration, you can replace this with WebSocket / API ticks."
    )

    col_l1, col_l2 = st.columns(2)

    with col_l1:
        ticker = "NIFTY"
        st.write(f"Ticker: **{ticker}**")

        if st.button("Simulate Tick"):
            # simple deterministic change based on time
            v = st.session_state.live_tick_value
            delta = (0.5 - (time.time() % 1)) * 4
            v += delta
            st.session_state.live_tick_value = v
            log(f"Live tick {ticker}: {v:.2f}", level="LIVE")

    with col_l2:
        st.metric("Simulated Value", f"{st.session_state.live_tick_value:.2f}")

    st.info("Each simulated tick is appended to logs with level 'LIVE'.")


# ---------------------------------------------------------------------
# TAB 5 – LOGS
# ---------------------------------------------------------------------
with tabs[4]:
    st.subheader("Logs")
    if st.session_state.logs:
        st.text_area(
            "Log Output",
            value="\n".join(st.session_state.logs),
            height=400,
        )
    else:
        st.info("No logs yet.")
