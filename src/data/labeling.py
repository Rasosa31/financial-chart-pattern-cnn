def label_window(prices, threshold=0.002):
    """
    prices: pandas Series con precios de cierre de la ventana
    threshold: retorno mínimo para evitar ruido
    """

    start_price = float(prices.iloc[0])
    end_price = float(prices.iloc[-1])

    ret = (end_price - start_price) / start_price

    if ret > threshold:
        return "up"
    elif ret < -threshold:
        return "down"
    else:
        return None
