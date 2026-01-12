import os
import pandas as pd
from src.data.download import download_data

# =========================
# CONFIG
# =========================
TICKER = "AAPL"
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
OUTPUT_PATH = "data/raw/prices.csv"

os.makedirs("data/raw", exist_ok=True)

# =========================
# DOWNLOAD DATA
# =========================
df = download_data(TICKER, START_DATE, END_DATE)

# =========================
# CLEAN & SAVE
# =========================
prices = df.reset_index()[["Date", "Open", "Close"]]
prices.columns = ["date", "open", "close"]

prices.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Raw prices saved to {OUTPUT_PATH}")
