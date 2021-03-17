import yfinance as yf
import numpy as np
import pandas as pd

# read data
amzn1 = pd.read_pickle("./amzn1.pkl")
amzn2 = pd.read_pickle("./amzn2.pkl")

spy1 = pd.read_pickle("./spy1.pkl")
spy2 = pd.read_pickle("./spy2.pkl")

vix1 = pd.read_pickle("./vix1.pkl")
vix2 = pd.read_pickle("./vix2.pkl")


# discard useless data
def clean_data(ticker, data):
    del data[0]
    temp = yf.Ticker(ticker)
    expiry = pd.Series(temp.options)
    expiry = pd.to_datetime(expiry)
    expiry = expiry - pd.Timestamp('today') < pd.Timedelta('80d')
    kept = len(expiry[expiry].index.values)
    newdata = data[0:kept]
    # add new columns of option type and expiry date
    def findType(s):
        if 'C' in s.replace(ticker, ''):
            return 'c'
        else:
            return 'p'
    def findDate(s):
        if 'VIXW' in s:
            return '20'+ s.replace('VIXW', '')[0:6]
        elif 'VIX' in s:
            return '20' + s.replace('VIX', '')[0:6]
        else:
            return '20' + s.replace(ticker, '')[0:6]
    for df in newdata:
        df['type'] = df['contractSymbol'].map(findType)
        df['expiry'] = df['contractSymbol'].map(findDate)
        df['expiry'] = pd.to_datetime(df['expiry'], infer_datetime_format=True)
        df['delta_t'] = (df['expiry'] - pd.Timestamp('today')) / np.timedelta64(1, 'Y')
    return newdata


amzn1 = clean_data("AMZN", amzn1)
amzn2 = clean_data("AMZN", amzn2)

spy1 = clean_data("SPY", spy1)
spy2 = clean_data("SPY", spy2)

vix1 = clean_data("^VIX", vix1)
vix2 = clean_data("^VIX", vix2)

DATA1 = [amzn1, spy1, vix1]
DATA2 = [amzn2, spy2, vix2]