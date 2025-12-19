import os, time, socket, ssl, multiprocessing as mp
from datetime import datetime
from tvDatafeed import TvDatafeed, Interval

# ==============================
# TradingView Credentials
# ==============================
USERNAME = "EGAVSIV"
PASSWORD = "Eric$1234"
tv = TvDatafeed(USERNAME, PASSWORD)

# ==============================
# Timeframes → Folder Mapping
# ==============================
TIMEFRAMES = {
    "D":  (Interval.in_daily,   "stock_data_D"),
    "W":  (Interval.in_weekly,  "stock_data_W"),
    "M":  (Interval.in_monthly, "stock_data_M"),
    "15": (Interval.in_15_minute, "stock_data_15"),
    "1H": (Interval.in_1_hour,  "stock_data_1H"),
}

BARS = 2000
RETRY_DELAY = 3
MAX_RETRY = 5

# ==============================
# Symbols (sample)
# ==============================
symbols = [
    "PIDILITIND",
    "PERSISTENT"
]

# ==============================
# Logs (repo root)
# ==============================
LOG_FILE = "download_log.txt"
ERROR_FILE = "error_symbols.txt"

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now()} | {msg}\n")
    print(msg)

def log_error(symbol, tf, err):
    with open(ERROR_FILE, "a") as f:
        f.write(f"{symbol},{tf},{err}\n")

# ==============================
# Fetch + Save
# ==============================
def fetch_save(args):
    symbol, tf_label, interval, folder = args
    os.makedirs(folder, exist_ok=True)
    attempt = 1

    while attempt <= MAX_RETRY:
        try:
            df = tv.get_hist(
                symbol=symbol,
                exchange="NSE",
                interval=interval,
                n_bars=BARS
            )

            if df is not None and not df.empty:
                df.to_parquet(os.path.join(folder, f"{symbol}.parquet"))
                log(f"[OK] {symbol} | TF:{tf_label}")
                return

            log(f"[EMPTY] {symbol} | TF:{tf_label} retry={attempt}")

        except Exception as e:
            msg = "Network error" if isinstance(e, (socket.timeout, ssl.SSLError)) else str(e)
            log(f"[ERROR] {symbol} | TF:{tf_label} | {msg}")

        attempt += 1
        time.sleep(RETRY_DELAY)

    log_error(symbol, tf_label, "Failed after retries")

# ==============================
# Runner
# ==============================
def run_all():
    log("===== DOWNLOAD STARTED =====")

    tasks = []
    for tf_label, (interval, folder) in TIMEFRAMES.items():
        for sym in symbols:
            tasks.append((sym, tf_label, interval, folder))

    workers = min(4, mp.cpu_count())
    with mp.Pool(workers) as pool:
        pool.map(fetch_save, tasks)

    log("===== DOWNLOAD FINISHED =====")

if __name__ == "__main__":
    run_all()
