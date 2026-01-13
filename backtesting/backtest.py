import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======================
# PARAMETERS
# ======================
THRESHOLD = 0.7
HOLD_DAYS = 120
COST_PER_TRADE = 0.001
SPLIT_DATE = "2021-01-01"

# ======================
# LOAD DATA
# ======================
preds = pd.read_csv("outputs/predictions.csv", parse_dates=["date"])
prices = pd.read_csv("data/raw/prices.csv", parse_dates=["date"])

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
df.dropna(inplace=True)

# ======================
# SIGNAL GENERATION
# ======================
df["raw_signal"] = (df["prob_up"] >= THRESHOLD).astype(int)

df["position"] = 0
hold_counter = 0

for i in range(len(df)):
    if hold_counter > 0:
        df.loc[i, "position"] = 1
        hold_counter -= 1
    elif df.loc[i, "raw_signal"] == 1:
        df.loc[i, "position"] = 1
        hold_counter = HOLD_DAYS - 1

df["trade"] = (
    (df["position"] == 1) &
    (df["position"].shift(1) == 0)
).astype(int)

df["strategy_return"] = (
    df["position"] * df["market_return"]
    - df["trade"] * COST_PER_TRADE
)

df.dropna(inplace=True)

# ======================
# SPLIT IS / OOS
# ======================
is_df = df[df["date"] < SPLIT_DATE].copy()
oos_df = df[df["date"] >= SPLIT_DATE].copy()

# ======================
# METRICS FUNCTION
# ======================
def compute_metrics(data, label):
    print(f"\n📊 {label} RESULTS")
    print("-" * 25)

    if len(data) == 0:
        print("No data / no trades in this period.")
        return None

    equity = (1 + data["strategy_return"]).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1

    sharpe = (
        data["strategy_return"].mean()
        / data["strategy_return"].std()
        * np.sqrt(252)
        if data["strategy_return"].std() != 0 else 0
    )

    print(f"Total Return : {equity.iloc[-1] - 1:.2%}")
    print(f"Sharpe Ratio : {sharpe:.2f}")
    print(f"Max Drawdown: {drawdown.min():.2%}")
    print(f"Trades      : {data['trade'].sum()}")

    return equity

# ======================
# RESULTS
# ======================
is_equity = compute_metrics(is_df, "IN-SAMPLE")
oos_equity = compute_metrics(oos_df, "OUT-OF-SAMPLE")

# ======================
# TRADE METRICS (OOS)
# ======================
if oos_df["trade"].sum() == 0:
    print("\n📈 OOS TRADE METRICS")
    print("-" * 25)
    print("No trades in Out-of-Sample period.")

# ======================
# PLOT
# ======================
plt.figure(figsize=(10, 5))

if is_equity is not None:
    plt.plot(is_df["date"], is_equity, label="In-Sample")

if oos_equity is not None:
    plt.plot(oos_df["date"], oos_equity, label="Out-of-Sample")

plt.title("Equity Curve (IS vs OOS)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
