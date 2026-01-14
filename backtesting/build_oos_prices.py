import yfinance as yf
import pandas as pd
from pathlib import Path

# ======================
# CONFIG
# ======================
TICKER = "SPY"
START_DATE = "2022-01-01"
END_DATE = "2022-06-30"
OUT_PATH = Path("data/test/prices.csv")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ======================
# DOWNLOAD
# ======================
print(f"Descargando precios para {TICKER}")
print(f"Desde {START_DATE} hasta {END_DATE}")

df = yf.download(
    TICKER,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True,
    progress=False
)

df = df.reset_index()[["Date", "Close"]]
df.columns = ["date", "close"]

df.to_csv(OUT_PATH, index=False)

print("✅ prices.csv generado correctamente")
print(df.head())
