import os, time, socket, ssl, multiprocessing as mp
from datetime import datetime
from tvDatafeed import TvDatafeed, Interval

USERNAME = "EGAVSIV"
PASSWORD = "Eric$1234"

INTERVAL = Interval.in_15_minute
FOLDER = "stock_data_15"


BARS = 4000
RETRY_DELAY = 3
MAX_RETRY = 5
WORKERS = 25

symbols = [
    # Paste symbols here
]

def fetch_save(symbol):
    os.makedirs(FOLDER, exist_ok=True)
    attempt = 1

    while attempt <= MAX_RETRY:
        try:
            tv = TvDatafeed(USERNAME, PASSWORD)

            df = tv.get_hist(
                symbol=symbol,
                exchange="NSE",
                interval=INTERVAL,
                n_bars=BARS
            )

            if df is not None and not df.empty:
                df.to_parquet(os.path.join(FOLDER, f"{symbol}.parquet"))
                print(f"[OK] {symbol} | TF:15")

                return

        except Exception as e:
            print(f"[ERROR] {symbol} | {e}")

        attempt += 1
        time.sleep(RETRY_DELAY)

if __name__ == "__main__":
    start = time.time()
    print("===== DAILY DOWNLOAD STARTED =====")

    with mp.Pool(WORKERS) as pool:
        pool.map(fetch_save, symbols)

    print("===== DAILY DOWNLOAD FINISHED =====")
    print("Time Taken:", round(time.time() - start, 2), "seconds")

