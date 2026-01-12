import yfinance as yf
import pandas as pd

def download_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)

    # 🔥 FIX CRÍTICO: eliminar MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 🔒 Asegurar tipo numérico
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    return df
