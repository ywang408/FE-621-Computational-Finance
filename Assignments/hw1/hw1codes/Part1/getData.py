import yfinance as yf
import pandas as pd
from datetime import datetime


def getData(ticker):
    temp = yf.Ticker(ticker)
    expiry = temp.options
    data = []
    for date in expiry:
        op = temp.option_chain(date)
        op_data = op.calls.append(op.puts)
        op_data['date'] = datetime.today().date()
        op_data['spotPrice'] = temp.history().iloc[-1, :]["Close"]
        data.append(op_data)
        pd.to_pickle(data, "./data.pkl")
    return data


def getMultiData(tickers):
    for ticker in tickers:
        filepath = "./" + ticker + ".pkl"
        temp = yf.Ticker(ticker)
        expiry = temp.options
        data = []
        for date in expiry:
            op = temp.option_chain(date)
            op_data = op.calls.append(op.puts)
            op_data['date'] = datetime.today().date()
            op_data['spotPrice'] = temp.history().iloc[-1, :]["Close"]
            data.append(op_data)
            pd.to_pickle(data, filepath)
    return


