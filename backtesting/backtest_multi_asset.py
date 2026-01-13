import pandas as pd
import numpy as np
from pathlib import Path

# ===============================
# CONFIG
# ===============================
DATA_DIR = Path("data/oos_multi_asset")  # carpeta con CSVs OOS
TOP_PCT = 0.02                            # top 2% señales
INITIAL_CAPITAL = 1.0

#Antes del rankin codigo tempora
print(symbol)
print(df["prob_up"].describe())
print("-" * 40)

# ===============================
# SIGNAL GENERATION (RANKING)
# ===============================
def generate_signals_by_rank(df, top_pct=0.02):
    """
    Generates trade signals using top percentile of prob_up
    """
    if df.empty:
        df["signal"] = False
        return df

    cutoff = df["prob_up"].quantile(1 - top_pct)
    df["signal"] = df["prob_up"] >= cutoff
    return df


# ===============================
# BACKTEST ENGINE
# ===============================
def run_backtest(df):
    """
    Simple long-only backtest.
    Entra cuando signal=True, sale cuando signal=False.
    """
    df = df.copy()
    df["position"] = df["signal"].astype(int)
    df["returns"] = df["close"].pct_change().fillna(0.0)
    df["strategy_returns"] = df["position"].shift(1).fillna(0) * df["returns"]
    df["equity"] = (1 + df["strategy_returns"]).cumprod()
    return df


# ===============================
# METRICS
# ===============================
def compute_metrics(df):
    if df.empty or df["equity"].empty:
        return None

    equity = df["equity"]
    total_return = equity.iloc[-1] - 1

    daily_ret = df["strategy_returns"]
    sharpe = np.sqrt(252) * daily_ret.mean() / daily_ret.std() if daily_ret.std() != 0 else 0.0

    rolling_max = equity.cummax()
    drawdown = (equity / rolling_max - 1).min()

    trades = (df["position"].diff() == 1).sum()

    return {
        "Total Return": total_return,
        "Sharpe": sharpe,
        "Max Drawdown": drawdown,
        "Trades": int(trades)
    }


# ===============================
# MAIN
# ===============================
def main():

    all_results = []

    for csv_file in DATA_DIR.glob("*.csv"):
        symbol = csv_file.stem

        df = pd.read_csv(csv_file, parse_dates=["date"])

        required_cols = {"date", "close", "prob_up"}
        if not required_cols.issubset(df.columns):
            print(f"⚠️ {symbol}: missing required columns")
            continue

        df = df.sort_values("date").reset_index(drop=True)

        # --- generate ranking-based signals ---
        df = generate_signals_by_rank(df, TOP_PCT)

        if df["signal"].sum() == 0:
            continue

        # --- run backtest ---
        df_bt = run_backtest(df)
        metrics = compute_metrics(df_bt)

        if metrics is None:
            continue

        metrics["Asset"] = symbol
        all_results.append(metrics)

    # ===============================
    # REPORT
    # ===============================
    print("\n📊 MULTI-ASSET OOS RESULTS")
    print("-------------------------")

    if not all_results:
        print("No trades across assets.")
        return

    results_df = pd.DataFrame(all_results).set_index("Asset")

    print(results_df.round(3))
    print("\nAverages:")
    print(results_df.mean().round(3))


if __name__ == "__main__":
    main()
