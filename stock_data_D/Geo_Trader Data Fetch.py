import pandas as pd
import os, random, time, logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tvDatafeed import TvDatafeed, Interval

# =====================================================
# CONFIG
# =====================================================
UPDATE_INTERVAL_SECONDS = 30
MAX_WORKERS = 16
BARS = 300

# =====================================================
# LOGGING
# =====================================================
logging.basicConfig(
    filename="tv_skip_reasons.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

# =====================================================
# BASE PATHS
# =====================================================
BASE_DIR = "market_data"

BUCKET_PATHS = {
    "broader_index": os.path.join(BASE_DIR, "broader_index"),
    "sector_index": os.path.join(BASE_DIR, "sector_index"),
    "fno": os.path.join(BASE_DIR, "fno"),
}

# 🔹 LOCAL FNO SOURCE FOLDERS
LOCAL_FNO_PATHS = {
    "15m": "stock_data_15",
    "1H": "stock_data_1H",
    "D": "stock_data_D",
    "W": "stock_data_W",
    "M": "stock_data_M",
}

# =====================================================
# TIMEFRAMES
# =====================================================
TIMEFRAMES = {
    "15m": Interval.in_15_minute,
    "1H": Interval.in_1_hour,
    "D": Interval.in_daily,
    "W": Interval.in_weekly,
    "M": Interval.in_monthly,
}

# Create folders
for path in BUCKET_PATHS.values():
    for tf in TIMEFRAMES.keys():
        os.makedirs(os.path.join(path, tf), exist_ok=True)

# =====================================================
# SYMBOL LISTS (same as yours)
# =====================================================
broader_index = [
    'NIFTY','BANKNIFTY','CNXMIDCAP','CNXSMALLCAP','CNX500',
    'CNXFINANCE','NIFTYJR','CNX100','NIFTY_TOP_10_EW'
]

sector_index = [
    'CNXREALTY','CNXPSUBANK','CNXMETAL','CNXIT','CNXSERVICE',
    'CNXPSE','CNXCONSUMPTION','CNXINFRA','CNXENERGY','CNXAUTO',
    'CNXFMCG','CNXPHARMA'
]

fno_symbols = [...]  # KEEP YOUR FULL LIST HERE

# =====================================================
# SYMBOL → BUCKET MAP
# =====================================================
SYMBOL_BUCKET = {}
for s in broader_index:
    SYMBOL_BUCKET[s] = "broader_index"
for s in sector_index:
    SYMBOL_BUCKET[s] = "sector_index"
for s in fno_symbols:
    SYMBOL_BUCKET[s] = "fno"

ALL_SYMBOLS = list(SYMBOL_BUCKET.keys())

# =====================================================
# INDICATORS (same as yours)
# =====================================================
def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_bollinger(close, period=20, std=2):
    mid = close.rolling(period).mean()
    dev = close.rolling(period).std()
    upper = mid + std * dev
    lower = mid - std * dev
    width = (upper - lower) / mid
    return upper, mid, lower, width


def calc_cpr(df):
    prev = df.shift(1)
    pivot = (prev['high'] + prev['low'] + prev['close']) / 3
    bc = (prev['high'] + prev['low']) / 2
    tc = (pivot * 2) - bc
    width = (tc - bc) / pivot
    return pivot, bc, tc, width


def calc_camarilla(df):
    prev = df.shift(1)
    rng = prev['high'] - prev['low']
    h3 = prev['close'] + rng * 1.1 / 4
    h4 = prev['close'] + rng * 1.1 / 2
    l3 = prev['close'] - rng * 1.1 / 4
    l4 = prev['close'] - rng * 1.1 / 2
    return h3, h4, l3, l4

# =====================================================
# WORKER FUNCTION
# =====================================================
def process_symbol(symbol):
    bucket = SYMBOL_BUCKET[symbol]
    base_path = BUCKET_PATHS[bucket]

    # =================================================
    # 🔵 BROADER + SECTOR → FETCH FROM TV
    # =================================================
    if bucket in ["broader_index", "sector_index"]:
        try:
            tv = TvDatafeed()
        except Exception as e:
            return f"❌ {symbol} | TV init failed | {e}"

        for tf, interval in TIMEFRAMES.items():
            try:
                df = tv.get_hist(
                    symbol=symbol,
                    exchange="NSE",
                    interval=interval,
                    n_bars=BARS
                )
            except:
                continue

            if df is None or df.empty:
                continue

            df = df.sort_index().tail(BARS)

            # Indicators
            df['rsi_14'] = calc_rsi(df['close'])
            df['bb_upper'], df['bb_middle'], df['bb_lower'], df['bb_width'] = \
                calc_bollinger(df['close'])
            df['cpr_pivot'], df['cpr_bc'], df['cpr_tc'], df['cpr_width'] = \
                calc_cpr(df)
            df['cam_h3'], df['cam_h4'], df['cam_l3'], df['cam_l4'] = \
                calc_camarilla(df)

            save_path = os.path.join(base_path, tf, f"{symbol}.parquet")
            df.to_parquet(save_path)

        return f"✅ {symbol} updated [TV]"

    # =================================================
    # 🟢 FNO → READ FROM LOCAL PARQUET
    # =================================================
    else:
        for tf in LOCAL_FNO_PATHS.keys():
            source_folder = LOCAL_FNO_PATHS[tf]
            source_file = os.path.join(source_folder, f"{symbol}.parquet")

            if not os.path.exists(source_file):
                continue

            df = pd.read_parquet(source_file)
            df = df.sort_index().tail(BARS)

            # Indicators
            df['rsi_14'] = calc_rsi(df['close'])
            df['bb_upper'], df['bb_middle'], df['bb_lower'], df['bb_width'] = \
                calc_bollinger(df['close'])
            df['cpr_pivot'], df['cpr_bc'], df['cpr_tc'], df['cpr_width'] = \
                calc_cpr(df)
            df['cam_h3'], df['cam_h4'], df['cam_l3'], df['cam_l4'] = \
                calc_camarilla(df)

            save_path = os.path.join(base_path, tf, f"{symbol}.parquet")
            df.to_parquet(save_path)

        return f"✅ {symbol} updated [LOCAL FNO]"


# =====================================================
# MAIN LOOP
# =====================================================
if __name__ == "__main__":
    print("\n🚀 Hybrid Collector Started (TV + Local FNO)\n")

    while True:
        start = time.time()
        symbols = ALL_SYMBOLS.copy()
        random.shuffle(symbols)

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_symbol, s) for s in symbols]
            for f in as_completed(futures):
                print(f.result())

        elapsed = int(time.time() - start)
        time.sleep(max(0, UPDATE_INTERVAL_SECONDS - elapsed))
