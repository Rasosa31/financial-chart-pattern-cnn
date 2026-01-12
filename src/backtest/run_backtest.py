import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
PREDICTIONS_CSV = "data/test/predictions.csv"
PRICES_CSV = "data/test/prices.csv"
INITIAL_CAPITAL = 10_000

# =========================
# LOAD DATA
# =========================
preds = pd.read_csv(PREDICTIONS_CSV, parse_dates=["date"])
prices = pd.read_csv(PRICES_CSV, parse_dates=["date"])

prices["date"] = prices["date"].dt.normalize()
preds["date"] = preds["date"].dt.normalize()


df = preds.merge(prices, on="date", how="inner")

if df.empty:
    raise ValueError("❌ Merge vacío: revisar alineación de fechas entre prices y predictions")

# Solo para prueba temporal

print("Prices rows:", len(prices))
print("Preds rows:", len(preds))
print("Merged rows:", len(df))
print(df.head())
#====================
# =========================
# STRATEGY LOGIC
# =========================
df["position"] = np.where(df["prediction"] == "up", 1, 0)

df["daily_return"] = df["close"].pct_change()
df["strategy_return"] = df["position"].shift(1) * df["daily_return"]

df["strategy_return"].fillna(0, inplace=True)

# =========================
# EQUITY CURVE
# =========================
df["equity"] = INITIAL_CAPITAL * (1 + df["strategy_return"]).cumprod()

# Buy & Hold
df["buy_hold"] = INITIAL_CAPITAL * (1 + df["daily_return"]).cumprod()

# =========================
# METRICS
# =========================
total_return = df["equity"].iloc[-1] / INITIAL_CAPITAL - 1
bh_return = df["buy_hold"].iloc[-1] / INITIAL_CAPITAL - 1

sharpe = (
    df["strategy_return"].mean() /
    df["strategy_return"].std()
) * np.sqrt(252)

print("\n📊 BACKTEST RESULTS")
print(f"Total Return (Strategy): {total_return:.2%}")
print(f"Total Return (Buy & Hold): {bh_return:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")

# =========================
# PLOT
# =========================
plt.figure(figsize=(10, 5))
plt.plot(df["date"], df["equity"], label="CNN Strategy")
plt.plot(df["date"], df["buy_hold"], label="Buy & Hold", linestyle="--")
plt.legend()
plt.title("Equity Curve - CNN Strategy vs Buy & Hold")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.grid()
plt.tight_layout()
plt.show()


