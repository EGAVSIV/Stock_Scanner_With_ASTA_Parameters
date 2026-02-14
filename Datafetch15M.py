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
    'SYNGENE','TATACONSUM','TATAELXSI','TATAPOWER','TATASTEEL','TATATECH','TCS','TECHM','TIINDIA',
    'TITAGARH','TITAN','TORNTPHARM','TORNTPOWER','TRENT','TVSMOTOR','ULTRACEMCO','UNIONBANK','UNITDSPR',
    'UNOMINDA','UPL','VBL','VEDL','VOLTAS','WIPRO','YESBANK','ZYDUSLIFE','BANKNIFTY','CNXFINANCE','CNXMIDCAP',
    'NIFTY','NIFTYJR','360ONE','ABB','ABCAPITAL','ADANIENSOL','ADANIENT','ADANIGREEN','ADANIPORTS','ALKEM',
    'AMBER','AMBUJACEM','ANGELONE','APLAPOLLO','APOLLOHOSP','ASHOKLEY','ASIANPAINT','ASTRAL','AUBANK',
    'AUROPHARMA','AXISBANK','BAJAJ_AUTO','BAJAJFINSV','BAJFINANCE','BANDHANBNK','BANKBARODA','BANKINDIA',
    'BDL','BEL','BHARATFORG','BHARTIARTL','BHEL','BIOCON','BLUESTARCO','BOSCHLTD','BPCL','BAJAJHLDNG','WAAREEENER','PREMIERENE','SWIGGY',"TMPV",
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

