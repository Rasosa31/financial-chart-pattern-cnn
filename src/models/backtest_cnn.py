import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
INITIAL_CAPITAL = 10_000
THRESHOLD = 0.5

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/test/prices.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Columns expected:
# date | open | close | y_prob

# =========================
# SIGNAL
# =========================
df["signal"] = (df["y_prob"] > THRESHOLD).astype(int)

# =========================
# RETURNS
# =========================
df["return"] = (df["close"] - df["open"]) / df["open"]

df["strategy_return"] = df["signal"] * df["return"]

# =========================
# EQUITY CURVE
# =========================
df["equity"] = INITIAL_CAPITAL * (1 + df["strategy_return"]).cumprod()

# =========================
# METRICS
# =========================
total_return = df["equity"].iloc[-1] / INITIAL_CAPITAL - 1
win_rate = (df["strategy_return"] > 0).mean()
max_dd = (
    df["equity"] / df["equity"].cummax() - 1
).min()

print("\nBACKTEST RESULTS")
print("====================")
print(f"Total Return: {total_return:.2%}")
print(f"Win Rate: {win_rate:.2%}")
print(f"Max Drawdown: {max_dd:.2%}")

# =========================
# PLOT
# =========================
plt.figure(figsize=(10, 5))
plt.plot(df["date"], df["equity"], label="CNN Strategy")
plt.title("Equity Curve - CNN Trading Strategy")
plt.xlabel("Date")
plt.ylabel("Equity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
