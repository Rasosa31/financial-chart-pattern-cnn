import pandas as pd

preds = pd.read_csv("outputs/predictions.csv")
print(len(preds), preds["date"].notna().sum())
