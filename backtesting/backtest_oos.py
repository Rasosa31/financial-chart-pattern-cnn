import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
PRED_PATH = "outputs/predictions.csv"
PRICE_PATH = "data/test/prices.csv"

THRESHOLD = 0.7        # señal fuerte
HOLD_DAYS = 10         # holding fijo
COST_PER_TRADE = 0.001 # 0.1%

# ======================
# LOAD DATA
# ======================
preds = pd.read_csv(PRED_PATH, parse_dates=["date"])
prices = pd.read_csv(PRICE_PATH, parse_dates=["date"])

# --- NORMALIZAR FECHAS (CLAVE) ---
preds["date"] = preds["date"].dt.normalize()
prices["date"] = prices["date"].dt.normalize()

# ======================
# MERGE OOS
# ======================
df = preds.merge(
    prices[["date", "close"]],
    on="date",
    how="inner"
).sort_values("date").reset_index(drop=True)

df["close"] = pd.to_numeric(df["close"], errors="coerce")
df = df.dropna(subset=["close"]).reset_index(drop=True)


# ======================
# DIAGNÓSTICO OOS
# ======================
print("\n--- DIAGNÓSTICO OOS ---")
print("Filas totales OOS:", len(df))

if df.empty:
    print("❌ DataFrame OOS vacío.")
    print("→ Las fechas de predictions y prices NO coinciden.")
    print("\nRango predictions:",
          preds["date"].min(), "→", preds["date"].max())
    print("Rango prices     :",
          prices["date"].min(), "→", prices["date"].max())
    print("----------------------")
    exit()

print("\nProbabilidades prob_up:")
print("Min:", df["prob_up"].min())
print("Max:", df["prob_up"].max())
print("----------------------")

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

print("\nRaw signals:")
print(df["raw_signal"].value_counts())
print("\nPosiciones:")
print(df["position"].value_counts())
print("----------------------\n")

# ======================
# RETURNS
# ======================
df["return"] = df["close"].pct_change().fillna(0.0)
df["trade"] = (df["position"].diff() == 1).astype(int)

df["strategy_return"] = (
    df["position"].shift(1).fillna(0) * df["return"]
    - df["trade"] * COST_PER_TRADE
)

df["equity"] = (1 + df["strategy_return"]).cumprod()

# ======================
# PROTECCIÓN: SIN TRADES
# ======================
if df["trade"].sum() == 0:
    print("⚠️ No se generaron operaciones en el periodo OOS.")
    print("Revisa threshold, señales o tamaño del OOS.")
    exit()

# ======================
# METRICS (PORTFOLIO)
# ======================
total_return = df["equity"].iloc[-1] - 1

daily_ret = df["strategy_return"]
sharpe = (
    np.sqrt(252) * daily_ret.mean() / daily_ret.std()
    if daily_ret.std() != 0 else 0.0
)

peak = df["equity"].cummax()
drawdown = (df["equity"] / peak - 1).min()

# ======================
# TRADE-LEVEL METRICS
# ======================
trades = []
in_trade = False

for i in range(len(df)):
    if df.loc[i, "trade"] == 1:
        entry_price = df.loc[i, "close"]
        entry_date = df.loc[i, "date"]
        in_trade = True

    if in_trade and (
        i == len(df) - 1 or df.loc[i + 1, "position"] == 0
    ):
        exit_price = df.loc[i, "close"]
        exit_date = df.loc[i, "date"]

        trade_return = exit_price / entry_price - 1
        duration = (exit_date - entry_date).days

        trades.append({
            "return": trade_return,
            "duration": duration
        })

        in_trade = False

trades_df = pd.DataFrame(trades)

# ======================
# REPORT
# ======================
print("\n📊 OOS BACKTEST RESULTS")
print("-------------------------")
print(f"Total Return : {total_return:.2%}")
print(f"Sharpe Ratio : {sharpe:.2f}")
print(f"Max Drawdown: {drawdown:.2%}")

print("\n📈 TRADE METRICS")
print("-------------------------")

print(f"Number of Trades    : {len(trades_df)}")
print(f"Win Rate            : {(trades_df['return'] > 0).mean():.2%}")
print(f"Avg Trade Return    : {trades_df['return'].mean():.2%}")
print(f"Best Trade          : {trades_df['return'].max():.2%}")
print(f"Worst Trade         : {trades_df['return'].min():.2%}")
print(f"Avg Duration (days) : {trades_df['duration'].mean():.1f}")

# ======================
# PLOT
# ======================
plt.figure(figsize=(10, 5))
plt.plot(df["date"], df["equity"], label="Strategy")
plt.title("OOS Equity Curve")
plt.xlabel("Date")
plt.ylabel("Equity")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
