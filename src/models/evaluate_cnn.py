from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# =========================
# Load model
# =========================
model = load_model("models/cnn_chart_model.h5")

# =========================
# Test data generator
# =========================
test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    "data/test/images",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# =========================
# Predictions
# =========================
# =========================
# Predictions
# =========================
y_prob = model.predict(test_data)
y_pred = (y_prob > 0.5).astype(int).flatten()
y_true = test_data.classes


# =========================
# Classification Report
# =========================
print("\nClassification Report (TEST):\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=["down", "up"]
))

# =========================
# Confusion Matrix
# =========================
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

labels = ["down", "up"]

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión - TEST")
plt.tight_layout()
plt.show()

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

thresholds = [0.4, 0.5, 0.55, 0.6, 0.65, 0.7]

for t in thresholds:
    print(f"\n===== Threshold: {t} =====")
    y_pred = (y_prob >= t).astype(int)

    print(classification_report(
        y_true,
        y_pred,
        target_names=["down", "up"],
        digits=3
    ))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
