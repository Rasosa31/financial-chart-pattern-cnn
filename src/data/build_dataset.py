import pandas as pd
from download import download_data
from windowing import sliding_windows
from labeling import label_window
from chart_gen import save_chart

WINDOW_SIZE = 60

df = download_data("AAPL", "2015-01-01", "2023-12-31")

windows = sliding_windows(df["Close"], WINDOW_SIZE)

for i, w in enumerate(windows):
    label = label_window(w)
    save_chart(
        df.iloc[i:i + WINDOW_SIZE],
        label,
        output_dir="data/images",
        idx=i
    )
