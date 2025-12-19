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
    'PIDILITIND','PERSISTENT','PETRONET','LTIM','INDIANB','INDHOTEL','HFCL','HAVELLS','BRITANNIA','BSE',
    'CAMS','CANBK','CDSL','CGPOWER','CHOLAFIN','CIPLA','COALINDIA','COFORGE','COLPAL','CONCOR','CROMPTON',
    'CUMMINSIND','CYIENT','DABUR','DALBHARAT','DELHIVERY','DIVISLAB','DIXON','DLF','DMART','DRREDDY',
    'EICHERMOT','ETERNAL','EXIDEIND','FEDERALBNK','FORTIS','GAIL','GLENMARK','GMRAIRPORT','GODREJCP','GODREJPROP',
    'GRASIM','HAL','HDFCAMC','HDFCBANK','HDFCLIFE','HEROMOTOCO','HINDALCO','HINDPETRO','HINDUNILVR','HINDZINC',
    'HUDCO','ICICIBANK','ICICIGI','ICICIPRULI','IDEA','IDFCFIRSTB','IEX','IGL','IIFL','INDIGO','INDUSINDBK',
    'INDUSTOWER','INFY','INOXWIND','IOC','IRCTC','IREDA','IRFC','ITC','JINDALSTEL','JIOFIN','JSWENERGY',
    'JSWSTEEL','JUBLFOOD','KALYANKJIL','KAYNES','KEI','KFINTECH','KOTAKBANK','KPITTECH','LAURUSLABS',
    'LICHSGFIN','LICI','LODHA','LT','LTF','LUPIN','M&M','MANAPPURAM','MANKIND','MARICO','MARUTI','MAXHEALTH',
    'MAZDOCK','MCX','MFSL','MOTHERSON','MPHASIS','MUTHOOTFIN','NATIONALUM','NAUKRI','NBCC','NCC','NESTLEIND',
    'NMDC','NTPC','NUVAMA','NYKAA','OBEROIRLTY','OFSS','OIL','ONGC','PAGEIND','PATANJALI','PAYTM',
    'PFC','PGEL','PHOENIXLTD','PIIND','PNB','PNBHOUSING','POLICYBZR','POLYCAB','NHPC','HCLTECH','POWERGRID',
    'PPLPHARMA','PRESTIGE','RBLBANK','RECLTD','RELIANCE','RVNL','SAIL','SAMMAANCAP','SBICARD','SBILIFE',
    'SBIN','SHREECEM','SHRIRAMFIN','SIEMENS','SOLARINDS','SONACOMS','SRF','SUNPHARMA','SUPREMEIND','SUZLON',
    'SYNGENE','TATACONSUM','TATAELXSI','TATAMOTORS','TATAPOWER','TATASTEEL','TATATECH','TCS','TECHM','TIINDIA',
    'TITAGARH','TITAN','TORNTPHARM','TORNTPOWER','TRENT','TVSMOTOR','ULTRACEMCO','UNIONBANK','UNITDSPR',
    'UNOMINDA','UPL','VBL','VEDL','VOLTAS','WIPRO','YESBANK','ZYDUSLIFE','BANKNIFTY','CNXFINANCE','CNXMIDCAP',
    'NIFTY','NIFTYJR','360ONE','ABB','ABCAPITAL','ADANIENSOL','ADANIENT','ADANIGREEN','ADANIPORTS','ALKEM',
    'AMBER','AMBUJACEM','ANGELONE','APLAPOLLO','APOLLOHOSP','ASHOKLEY','ASIANPAINT','ASTRAL','AUBANK',
    'AUROPHARMA','AXISBANK','BAJAJ_AUTO','BAJAJFINSV','BAJFINANCE','BANDHANBNK','BANKBARODA','BANKINDIA',
    'BDL','BEL','BHARATFORG','BHARTIARTL','BHEL','BIOCON','BLUESTARCO','BOSCHLTD','BPCL','BAJAJHLDNG','WAAREEENER','PREMIERENE','SWIGGY'
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

