import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======================
# PARAMETERS
# ======================
COST_PER_TRADE = 0.001  # 0.1% por trade

# ======================
# LOAD DATA
# ======================
preds = pd.read_csv("outputs/predictions.csv")
prices = pd.read_csv("data/raw/prices.csv")

preds["date"] = pd.to_datetime(preds["date"])
prices["date"] = pd.to_datetime(prices["date"])

# ======================
# ALIGN DATA
# ======================
df = preds.merge(
    prices[["date", "close"]],
    on="date",
    how="inner"
).sort_values("date").reset_index(drop=True)

# ======================
# RETURNS
# ======================
df["next_close"] = df["close"].shift(-1)
df["market_return"] = df["next_close"] / df["close"] - 1


#===============
# ======================
# THRESHOLD OPTIMIZATION
# ======================

results = []

thresholds = np.arange(0.50, 0.81, 0.05)

for THRESHOLD in thresholds:

    temp = df[["date", "prob_up", "market_return"]].copy()

    # Position based on threshold
    temp["position"] = np.where(temp["prob_up"] >= THRESHOLD, 1, 0)

    # Trading costs (entry/exit)
    # Trading costs (ONLY entries: 0 -> 1)
    temp["trade"] = (
        (temp["position"] == 1) &
        (temp["position"].shift(1) == 0)
    ).astype(int)

    transaction_cost = 0.001  # 0.1%

    
    temp["strategy_return"] = (
        temp["position"] * temp["market_return"]
        - temp["trade"] * transaction_cost
    )

    temp = temp.dropna().reset_index(drop=True)

    # Equity
    temp["equity"] = (1 + temp["strategy_return"]).cumprod()
    temp["peak"] = temp["equity"].cummax()
    temp["drawdown"] = temp["equity"] / temp["peak"] - 1

    total_return = temp["equity"].iloc[-1] - 1
    max_dd = temp["drawdown"].min()

    sharpe = (
        temp["strategy_return"].mean()
        / temp["strategy_return"].std()
        * np.sqrt(252)
        if temp["strategy_return"].std() != 0 else 0
    )

    trades = temp["trade"].sum()

    results.append({
        "threshold": THRESHOLD,
        "return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "trades": trades
    })

results_df = pd.DataFrame(results)

print("\n📊 THRESHOLD OPTIMIZATION RESULTS")
print(results_df.round(3))

#===============
# ======================
# STRATEGY (probability filter)
# ======================
THRESHOLD = 0.9

df["position"] = np.where(df["prob_up"] >= THRESHOLD, 1, 0)
df["trade"] = df["position"].diff().abs().fillna(0)
df["strategy_return"] = (
    df["position"] * df["market_return"]
    - df["trade"] * COST_PER_TRADE
)

df = df.dropna().reset_index(drop=True)

# --- SHARPE RATIO ---
daily_mean = df["strategy_return"].mean()
daily_std = df["strategy_return"].std()

sharpe_ratio = (daily_mean / daily_std) * (252 ** 0.5)


# ======================
# EQUITY CURVES
# ======================
df["market_equity"] = (1 + df["market_return"]).cumprod()
df["strategy_equity"] = (1 + df["strategy_return"]).cumprod()

# ======================
# DRAWDOWN
# ======================
df["strategy_peak"] = df["strategy_equity"].cummax()
df["strategy_drawdown"] = df["strategy_equity"] / df["strategy_peak"] - 1
max_drawdown = df["strategy_drawdown"].min()

# ======================
# METRICS
# ======================
total_strategy_return = df["strategy_equity"].iloc[-1] - 1
total_market_return = df["market_equity"].iloc[-1] - 1
hit_rate = (df["strategy_return"] > 0).mean()
trades = df["position"].sum()

print("\n📊 BACKTEST RESULTS")
print("------------------")
print(f"Total Strategy Return: {total_strategy_return:.2%}")
print(f"Total Market Return : {total_market_return:.2%}")
print(f"Hit Rate            : {hit_rate:.2%}")
print(f"Trades              : {trades}")
print(f"Max Drawdown        : {max_drawdown:.2%}")
print(f"Sharpe Ratio        : {sharpe_ratio:.2f}")

# ======================
# PLOT EQUITY CURVE
# ======================
plt.figure(figsize=(10, 5))
plt.plot(df["date"], df["strategy_equity"], label="Strategy")
plt.plot(df["date"], df["market_equity"], label="Buy & Hold", linestyle="--")

plt.title("Equity Curve")
plt.xlabel("Date")
plt.ylabel("Equity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
