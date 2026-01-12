import os
import mplfinance as mpf
import pandas as pd

def save_chart(df, label, output_dir, idx):
    os.makedirs(f"{output_dir}/{label}", exist_ok=True)

    # --- 1. Ensure proper columns (flatten if MultiIndex) ---
    df = df.copy()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[required_cols]

    # --- 2. Ensure DatetimeIndex ---
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # --- 3. Force numeric conversion (VERY important) ---
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- 4. Drop NaNs (mplfinance is strict) ---
    df.dropna(inplace=True)

    # --- 5. Safety check ---
    if len(df) < 10:
        return

    file_path = f"{output_dir}/{label}/{label}_{idx}.png"

    mpf.plot(
        df,
        type="candle",
        style="charles",
        volume=False,
        savefig=file_path
    )
