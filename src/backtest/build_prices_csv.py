import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =========================
# CONFIG
# =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_PATH = "models/cnn_chart_model.h5"
OUTPUT_PATH = "data/test/prices.csv"

# ⚠️ ajusta esto según tu dataset original
RAW_PRICE_CSV = "data/raw/prices.csv"
WINDOW_SIZE = 30

# =========================
# LOAD MODEL
# =========================
model = load_model(MODEL_PATH)

# =========================
# LOAD TEST IMAGES
# =========================
gen = ImageDataGenerator(rescale=1./255)

test_data = gen.flow_from_directory(
    "data/test/images",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# =========================
# PREDICTIONS
# =========================
y_prob = model.predict(test_data).flatten()

# =========================
# LOAD RAW PRICES
# =========================
prices = pd.read_csv(RAW_PRICE_CSV, parse_dates=["date"])
prices = prices.sort_values("date").reset_index(drop=True)

# =========================
# BUILD WINDOWS
# =========================
rows = []

for i, prob in enumerate(y_prob):
    start = i
    end = i + WINDOW_SIZE - 1

    if end >= len(prices):
        break

    open_price = prices.loc[start, "close"]
    close_price = prices.loc[end, "close"]
    date = prices.loc[end, "date"]

    rows.append({
        "date": date,
        "open": open_price,
        "close": close_price,
        "y_prob": prob
    })

df = pd.DataFrame(rows)

# =========================
# SAVE
# =========================
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ prices.csv creado en {OUTPUT_PATH}")
