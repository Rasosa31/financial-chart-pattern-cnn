import os
from collections import Counter

BASE_DIR = "data/images"

counts = {}

for label in os.listdir(BASE_DIR):
    label_path = os.path.join(BASE_DIR, label)
    if os.path.isdir(label_path):
        counts[label] = len(os.listdir(label_path))

print("Class distribution:")
for k, v in counts.items():
    print(f"{k}: {v}")
