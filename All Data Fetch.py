import os, time, socket, ssl, traceback, multiprocessing as mp
from datetime import datetime
from tvDatafeed import TvDatafeed, Interval

# =====================================================
#    USER CREDENTIALS (As requested, kept in script)
# =====================================================
USERNAME = "EGAVSIV"
PASSWORD = "Eric$1234"
tv = TvDatafeed(USERNAME, PASSWORD)

# =====================================================
#    TIMEFRAME CONFIG
# =====================================================
TIMEFRAMES = {
    "D":  Interval.in_daily,
    "W":  Interval.in_weekly,
    "M":  Interval.in_monthly,
    "15": Interval.in_15_minute,
    "1H": Interval.in_1_hour
}

BARS = 2000
RETRY_DELAY = 3
MAX_RETRY = 5

# =====================================================
#    SYMBOL LIST (Master List)
# =====================================================
symbols = [
    'PIDILITIND','PERSISTENT'
]

# =====================================================
#    Logging Setup
# =====================================================
LOG_FILE = "download_log.txt"
ERROR_FILE = "error_symbols.txt"

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now()}  |  {msg}\n")
    print(msg)

def log_error(symbol, timeframe, error):
    with open(ERROR_FILE, "a") as f:
        f.write(f"{symbol},{timeframe},{error}\n")

# =====================================================
#    FETCH FUNCTION FOR MULTIPROCESSING
# =====================================================
def fetch_save(args):
    symbol, tf_label, interval = args
    attempt = 1

    output_dir = f"stock_data_{tf_label}"
    os.makedirs(output_dir, exist_ok=True)

    while attempt <= MAX_RETRY:
        try:
            df = tv.get_hist(
                symbol=symbol,
                exchange="NSE",
                interval=interval,
                n_bars=BARS
            )

            if df is not None and not df.empty:
                df["timeframe"] = tf_label
                df.to_parquet(os.path.join(output_dir, f"{symbol}.parquet"))
                log(f"[OK] {symbol:<12} | TF:{tf_label}")
                return

            log(f"[WARNING] empty data {symbol} TF:{tf_label} retry={attempt}")

        except Exception as e:
            msg = str(e)
            if isinstance(e, (socket.timeout, ssl.SSLError)):
                msg = "Network Timeout"
            log(f"[TIMEOUT] {symbol} TF:{tf_label} retry={attempt} Error:{msg}")

        attempt += 1
        time.sleep(RETRY_DELAY)

    log(f"[FAILED] {symbol:<12} after retries | TF:{tf_label}")
    log_error(symbol, tf_label, "Failed after retries")

# =====================================================
#    MASTER EXECUTION PARALLEL
# =====================================================
def run_all():
    start = time.time()
    log("===== DOWNLOAD STARTED =====")

    tasks = []
    for tf_label, interval in TIMEFRAMES.items():
        for symbol in symbols:
            tasks.append((symbol, tf_label, interval))

    # CPU parallel workers
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(fetch_save, tasks)

    log("\n===== DOWNLOAD FINISHED =====")
    log(f"Time taken: {round(time.time()-start,2)} seconds")

# =====================================================
#    ENTRY POINT
# =====================================================
if __name__ == "__main__":
    run_all()


