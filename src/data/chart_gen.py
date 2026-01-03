import os
import mplfinance as mpf

def save_chart(prices, label, output_dir, idx):
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    file_path = os.path.join(label_dir, f"chart_{idx}.png")

    mpf.plot(
        prices,
        type="candle",
        style="charles",
        volume=False,
        savefig=dict(fname=file_path, dpi=100)
    )
