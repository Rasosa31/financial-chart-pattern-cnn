import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

BASE_DIR = "data/walk_forward"


def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

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

    return model


results = []

for split in sorted(os.listdir(BASE_DIR)):
    split_path = os.path.join(BASE_DIR, split)

    train_dir = os.path.join(split_path, "train", "images")
    test_dir  = os.path.join(split_path, "test", "images")

    print(f"\n🚀 Entrenando {split}")

    train_gen = ImageDataGenerator(rescale=1./255)
    test_gen  = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True
    )

    test_data = test_gen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

    model = build_model()

    model.fit(
        train_data,
        epochs=EPOCHS,
        verbose=1
    )

    preds = model.predict(test_data)
    preds = (preds > 0.5).astype(int).ravel()

    report = classification_report(
        test_data.classes,
        preds,
        target_names=["down", "up"],
        output_dict=True
    )

    results.append({
        "split": split,
        "accuracy": report["accuracy"],
        "f1_down": report["down"]["f1-score"],
        "f1_up": report["up"]["f1-score"],
    })

# ---- resumen final ----
df = pd.DataFrame(results)
print("\n📊 RESULTADOS WALK-FORWARD\n")
print(df)

df.to_csv("walk_forward_results.csv", index=False)
