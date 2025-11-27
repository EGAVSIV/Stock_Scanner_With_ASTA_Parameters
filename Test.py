import os, glob
import pandas as pd
import numpy as np
import talib
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

#####################################################################
# CONFIG
#####################################################################

FOLDERS = {
    "D":   "stock_data_D",
    "W":   "stock_data_W",
    "M":   "stock_data_M",
    "15m": "stock_data_15",
    "1h":  "stock_data_1H",
}

FILTER_OPTIONS = [
    "MACD uptick","MACD downtick","MACD > 0","MACD < 0",
    "MACD PCO","MACD NCO","51326_PCO","51326_NCO",
    "UBBC","LBBC","Price > Med","Price < Med",
    "Ungli","DI Bull","DI Bear","Stoch PCO","Stoch NCO"
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

UI_COLORS = [
    "#3CB371","#DC143C",
    "#1E90FF","#FF8C00",
    "#2ECC71","#E74C3C",
    "#3498DB","#F39C12"
]

#####################################################################
# INDICATOR CALCULATION
#####################################################################

def compute_indicators(df):
    close=df.close.astype(float);high=df.high.astype(float);low=df.low.astype(float)
    df["ema5"]=talib.EMA(close,5)
    df["ema13"]=talib.EMA(close,13)
    df["ema20"]=talib.EMA(close,20)
    df["ema26"]=talib.EMA(close,26)
    df["ema50"]=talib.EMA(close,50)

    df["rsi"]=talib.RSI(close,14)

    df["macd"],df["signal"],df["hist"]=talib.MACD(close)

    df["bb_up"],df["bb_mid"],df["bb_low"]=talib.BBANDS(close)

    k,d=talib.STOCH(high,low,close)
    df["k"]=k;df["d"]=d

    df["adx"]=talib.ADX(high,low,close)
    df["plus_di"]=talib.PLUS_DI(high,low,close)
    df["minus_di"]=talib.MINUS_DI(high,low,close)

    return df.dropna()

#####################################################################
# FLAGS / ABBREVIATIONS
#####################################################################

def add_flags(df):
    df["macd_uptick"]=df.macd>df.macd.shift(1)
    df["macd_downtick"]=df.macd<df.macd.shift(1)
    df["macd_pos"]=df.macd>0
    df["macd_neg"]=df.macd<0
    df["macd_pco"]=df.macd>df.signal
    df["macd_nco"]=df.macd<df.signal

    df["ema_51326_pco"]=(df.ema5>df.ema13)&(df.ema13>df.ema26)
    df["ema_51326_nco"]=(df.ema5<df.ema13)&(df.ema13<df.ema26)

    df["ubb_c"]=df.bb_up>df.bb_up.shift(1)
    df["lbbc_c"]=df.bb_low<df.bb_low.shift(1)

    df["price_gt_med"]=df.close>df.bb_mid
    df["price_lt_med"]=df.close<df.bb_mid

    df["ungli"]=(df.adx>14)&(df.adx>df.adx.shift(1))&(df.adx.shift(1)<df.adx.shift(2))

    df["di_bull"]=df.plus_di>df.minus_di
    df["di_bear"]=df.plus_di<df.minus_di

    df["stoch_pco"]=df.k>df.d
    df["stoch_nco"]=df.k<df.d
    return df

#####################################################################
# LOAD LAST ROW PER SYMBOL
#####################################################################

def load_latest(folder):
    rows=[]
    if not os.path.isdir(folder):
        return pd.DataFrame()
    for file in glob.glob(os.path.join(folder,"*.parquet")):
        df=pd.read_parquet(file).sort_index()
        df=compute_indicators(df)
        df=add_flags(df)
        rows.append(df.iloc[-1])
    df=pd.DataFrame(rows)
    df.set_index("symbol",inplace=True)
    return df

#####################################################################
# MAIN SCAN
#####################################################################

def run_scan(tf1,tf2,t1_filters,t2_filters,r1_cond,r1_val,r2_cond,r2_val):
    df1=load_latest(FOLDERS[tf1]).add_suffix("_"+tf1)
    df2=load_latest(FOLDERS[tf2]).add_suffix("_"+tf2)
    merged=df1.join(df2,how="inner")

    for x in t1_filters:
        if x!="None":
            col=f"{FILTER_COLUMN_MAP[x]}_{tf1}"
            merged=merged[merged[col]==True]

    for x in t2_filters:
        if x!="None":
            col=f"{FILTER_COLUMN_MAP[x]}_{tf2}"
            merged=merged[merged[col]==True]

    # RSI TF1
    if r1_cond!="None":
        col=f"rsi_{tf1}";val=float(r1_val)
        merged=merged[merged[col]>=val] if r1_cond=="RSI >=" else merged[merged[col]<=val]

    # RSI TF2
    if r2_cond!="None":
        col=f"rsi_{tf2}";val=float(r2_val)
        merged=merged[merged[col]>=val] if r2_cond=="RSI >=" else merged[merged[col]<=val]

    return merged

#####################################################################
# GUI
#####################################################################

def launch():

    root=tk.Tk()
    root.title("Multi-TF Condition Scanner By GS Yadav")
    root.geometry("1700x950")
    root.configure(bg="#101010")

    style=ttk.Style()
    style.theme_use("clam")

    #################################################################
    # TITLE + BRANDING
    #################################################################

    tk.Label(root,text="Multi-TF Multi-Condition Filter By GS",
             fg="#F39C12",bg="#101010",
             font=("Segoe UI Black",24,"bold")).pack(pady=10)

    tk.Label(root,text="Designed by GS Yadav",
             fg="#3498DB",bg="#101010",
             font=("Georgia",20,"bold")).place(x=1450,y=20)

    #################################################################
    # TOP FRAME
    #################################################################

    top=tk.Frame(root,bg="#101010")
    top.pack()

    ttk.Label(top,text="TF1").grid(row=0,column=0)
    tf1=tk.StringVar(value="D")
    ttk.Combobox(top,textvariable=tf1,values=list(FOLDERS.keys()),width=5).grid(row=0,column=1)

    ttk.Label(top,text="TF2").grid(row=0,column=2)
    tf2=tk.StringVar(value="W")
    ttk.Combobox(top,textvariable=tf2,values=list(FOLDERS.keys()),width=5).grid(row=0,column=3)

    # RSI TF1
    ttk.Label(top,text="RSI TF1").grid(row=0,column=4)
    r1=tk.StringVar(value="None")
    ttk.Combobox(top,textvariable=r1,values=["None","RSI >=","RSI <="],width=7).grid(row=0,column=5)
    r1v=tk.StringVar()
    ttk.Entry(top,textvariable=r1v,width=5).grid(row=0,column=6)

    # RSI TF2
    ttk.Label(top,text="RSI TF2").grid(row=0,column=7)
    r2=tk.StringVar(value="None")
    ttk.Combobox(top,textvariable=r2,values=["None","RSI >=","RSI <="],width=7).grid(row=0,column=8)
    r2v=tk.StringVar()
    ttk.Entry(top,textvariable=r2v,width=5).grid(row=0,column=9)

    #################################################################
    # CONDITIONS PANELS
    #################################################################

    cond_frame=tk.Frame(root,bg="#101010")
    cond_frame.pack(pady=10)

    tf1_vars=[]; tf2_vars=[]

    frame1=tk.LabelFrame(cond_frame,text="TF1 Conditions",bg="#101010",fg="white")
    frame1.pack(side=tk.LEFT,padx=10)
    for i in range(5):
        v=tk.StringVar(value="None")
        ttk.Combobox(frame1,textvariable=v,values=["None"]+FILTER_OPTIONS,width=15).pack()
        tf1_vars.append(v)

    frame2=tk.LabelFrame(cond_frame,text="TF2 Conditions",bg="#101010",fg="white")
    frame2.pack(side=tk.LEFT,padx=10)
    for i in range(5):
        v=tk.StringVar(value="None")
        ttk.Combobox(frame2,textvariable=v,values=["None"]+FILTER_OPTIONS,width=15).pack()
        tf2_vars.append(v)

    #################################################################
    # RESULT TABLE
    #################################################################

    table_frame=tk.Frame(root)
    table_frame.pack(fill=tk.BOTH,expand=True)

    tree=ttk.Treeview(table_frame)
    vs=ttk.Scrollbar(table_frame,orient="vertical",command=tree.yview)
    tree.configure(yscrollcommand=vs.set)
    tree.pack(side=tk.LEFT,fill=tk.BOTH,expand=True)
    vs.pack(side=tk.RIGHT,fill=tk.Y)

    style.configure("Treeview",
                    background="white",
                    foreground="black",
                    rowheight=25,
                    font=("Calibri",12))
    style.configure("Treeview.Heading",
                    font=("Calibri",13,"bold"),
                    background="#3498DB",
                    foreground="white")

    lbl=tk.Label(root,text="",fg="yellow",bg="#101010",font=("Calibri",14))
    lbl.pack()

    #################################################################
    # BUTTON ACTIONS
    #################################################################

    def run_click():
        df=run_scan(tf1.get(),tf2.get(),
                    [v.get() for v in tf1_vars],
                    [v.get() for v in tf2_vars],
                    r1.get(),r1v.get(),r2.get(),r2v.get())

        for r in tree.get_children():tree.delete(r)

        if df.empty:
            lbl.config(text="No Stocks Found")
            return

        df=df.reset_index()

        cols=["symbol",
              f"close_{tf1.get()}",
              f"close_{tf2.get()}"]

        tree["columns"]=cols;tree["show"]="headings"

        for c in cols:
            tree.heading(c,text=c)
            tree.column(c,width=180,anchor="center")

        for _,row in df.iterrows():
            tree.insert("",tk.END,values=[row[c] for c in cols])

        lbl.config(text=f"Total Stocks: {len(df)}")

    def reset():
        tf1.set("D"); tf2.set("W")
        r1.set("None"); r1v.set("")
        r2.set("None"); r2v.set("")
        for v in tf1_vars: v.set("None")
        for v in tf2_vars: v.set("None")
        for row in tree.get_children(): tree.delete(row)
        lbl.config(text="")

    def save_csv():
        f=filedialog.asksaveasfilename(defaultextension=".csv")
        if f:
            df=run_scan(tf1.get(),tf2.get(),
                        [v.get() for v in tf1_vars],
                        [v.get() for v in tf2_vars],
                        r1.get(),r1v.get(),r2.get(),r2v.get())
            df.to_csv(f)

    #################################################################
    # BUTTONS
    #################################################################

    tk.Button(root,text="SCAN",bg="#3CB371",fg="black",
              font=("Calibri",16,"bol{}".format('d')),
              command=run_click).pack(pady=10)

    tk.Button(root,text="RESET FILTERS",bg="#DC143C",fg="white",
              font=("Calibri",14,"bold"),
              command=reset).pack(pady=5)

    tk.Button(root,text="DOWNLOAD CSV",
              bg="#3498DB",fg="white",
              font=("Calibri",14,"bold"),
              command=save_csv).pack(pady=5)

    root.mainloop()

#####################################################################
if __name__=="__main__":
    launch()
