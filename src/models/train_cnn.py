import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.utils.class_weight import compute_class_weight

# ------------------------
# CONFIG
# ------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

TRAIN_DIR = "data/train"
VAL_DIR   = "data/val"

# ------------------------
# DATA GENERATORS
# ------------------------
train_gen = ImageDataGenerator(
    rescale=1.0 / 255
)

val_gen = ImageDataGenerator(
    rescale=1.0 / 255
)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True
)

val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# ------------------------
# CLASS WEIGHTS
# ------------------------
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weight = dict(enumerate(class_weights_array))

print("\nClass mapping:", train_data.class_indices)
print("Class weights:", class_weight)

# ------------------------
# MODEL
# ------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    MaxPooling2D(),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ------------------------
# TRAIN
# ------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weight
)

# ------------------------
# SAVE MODEL
# ------------------------
os.makedirs("models", exist_ok=True)
model.save("models/cnn_chart_model.keras")

print("\n✅ Modelo guardado en models/cnn_chart_model.keras")
