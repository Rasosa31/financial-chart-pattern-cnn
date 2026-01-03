import pandas as pd

def label_window(prices, threshold=0.02):
    start_price = prices.iloc[0]
    end_price = prices.iloc[-1]

    ret = (end_price - start_price) / start_price

    if ret >= threshold:
        return "bullish"
    elif ret <= -threshold:
        return "bearish"
    else:
        return "sideways"
