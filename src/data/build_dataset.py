import os
from src.data.download import download_data
from src.data.windowing import sliding_windows
from src.data.labeling import label_window
from src.data.chart_gen import save_chart

WINDOW_SIZE = 60
TICKER = "AAPL"

# --- splits temporales (SIN LEAKAGE) ---
SPLITS = {
    "train": ("2015-01-01", "2020-12-31"),
    "val":   ("2021-01-01", "2022-12-31"),
    "test":  ("2023-01-01", "2024-12-31"),
}

for split, (start, end) in SPLITS.items():
    print(f"\n📦 Building {split} dataset...")

    df = download_data(TICKER, start, end)
    prices = df["Close"]

    windows = sliding_windows(prices, WINDOW_SIZE)

    output_dir = f"data/{split}/images"
    os.makedirs(output_dir, exist_ok=True)

    for i, window in enumerate(windows):
        label = label_window(window)

        if label not in ("up", "down"):
            continue

        try:
            save_chart(
                df.iloc[i:i + WINDOW_SIZE],
                label,
                output_dir=output_dir,
                idx=i
            )
        except Exception as e:
            print(f"⚠️ Skipping window {i}: {e}")
