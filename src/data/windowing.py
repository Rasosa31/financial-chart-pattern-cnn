def sliding_windows(series, window_size):
    windows = []
    for i in range(len(series) - window_size):
        window = series.iloc[i:i + window_size]
        windows.append(window)
    return windows
