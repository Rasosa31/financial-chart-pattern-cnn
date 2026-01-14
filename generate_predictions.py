import os
import re
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ======================
# CONFIG
# ======================
MODEL_PATH = "models/cnn_chart_model.keras"
TEST_DIR = "data/test/images"
OUTPUT_PATH = "outputs/predictions.csv"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ======================
# LOAD MODEL
# ======================
print("Model exists:", os.path.exists(MODEL_PATH))
model = load_model(MODEL_PATH)

# ======================
# LOAD PRICES (ground truth timeline)
# ======================
prices = pd.read_csv("data/test/prices.csv")
prices["date"] = pd.to_datetime(prices["date"])
prices = prices.reset_index(drop=True)  # CRÍTICO

# ======================
# IMAGE GENERATOR (NO SHUFFLE!)
# ======================
datagen = ImageDataGenerator(rescale=1.0 / 255)

test_gen = datagen.flow_from_directory(
    directory=TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode=None
)

# ======================
# PREDICT
# ======================
pred_probs = model.predict(test_gen).ravel()

# ======================
# EXTRACT FILENAMES
# ======================
filenames = [os.path.basename(f) for f in test_gen.filenames]

# ======================
# EXTRACT INDEX FROM FILENAME
# ======================
def extract_index(filename):
    match = re.search(r"_(\d+)\.png", filename)
    if not match:
        raise ValueError(f"No index found in filename: {filename}")
    return int(match.group(1))

indices = [extract_index(f) for f in filenames]

max_valid_index = len(prices) - 1
valid_rows = [
    (i, prob) for i, prob in zip(indices, pred_probs)
    if i <= max_valid_index
]

dates = [prices.iloc[i]["date"] for i, _ in valid_rows]
probs = [prob for _, prob in valid_rows]


# ======================
# BUILD FINAL DATAFRAME
# ======================
df = pd.DataFrame({
    "date": dates,
    "prob_up": probs
})

df["prediction"] = np.where(df["prob_up"] >= 0.5, "up", "down")

# ======================
# SORT & SAVE
# ======================
df = df.sort_values("date").reset_index(drop=True)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("✅ predictions.csv generated successfully")
print(df.head())
