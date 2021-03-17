from Part1.cleanData import DATA1, DATA2
from root_methods import *
from BS_formula import *
import pandas as pd


def get_impliedVol(Type, S, K, T, r, P, method):
    def price_diff(sigma):
        return BS_formula(Type, S, K, T, sigma, r) - P
    if method == 'b':
        return bisection(price_diff, 0.001, 1)
    elif method == 'n':
        price_diff_prime = lambda x: vega(S, K, T, x, r)
        return newton_method(price_diff, price_diff_prime, 0.5)


def moneyness(S, K):
    if S/K < 0.95:
        return 'inTheMoney'
    elif S/K > 1.05:
        return 'outTheMoney'
    else:
        return 'atTheMoney'


def parity(Type, S, K, T, r, P):
    if Type == 'c':
        return P - S + K * np.exp(-r*T)
    elif Type == 'p':
        return P + S - K * np.exp(-r*T)

# calculate (bid+ask)/2
for i in range(3):
    for df in DATA1[i]:
        df['optionPrice'] = df['bid']/2 + df['ask']/2

# calculate implied vol using bisection
r = 0.07/100
now = pd.Timestamp.now()
for i in range(2):
    for df in DATA1[i]:
        df['bisec_Root'] = df.apply(lambda x: get_impliedVol(x.type, x.spotPrice, x.strike,
                                                             x.delta_t, r, x.optionPrice, 'b'), axis=1)
end = pd.Timestamp.now()
bisec_time = (end - now) / np.timedelta64(1, 's')
print("It takes {} seconds".format(bisec_time))

# calculate implied vol using newton's method
now = pd.Timestamp.now()
for i in range(2):
    for df in DATA1[i]:
        df['newton_Root'] = df.apply(lambda x: get_impliedVol(x.type, x.spotPrice, x.strike,
                                                             x.delta_t, r, x.optionPrice, 'n'), axis=1)
end = pd.Timestamp.now()
newton_time = (end - now) / np.timedelta64(1, 's')
print("It takes {} seconds".format(newton_time))

# determine moneyness
for i in range(3):
    for df in DATA1[i]:
        df['moneyness'] = df.apply(lambda x: moneyness(x.spotPrice, x.strike), axis=1)

# calculate average vol
# amzn
expiry_date = []
in_mean = []
out_mean = []
at_mean = []
for df in DATA1[0]:
    expiry_date.append(df.expiry.iloc[0])
    temp = df[df.moneyness == 'atTheMoney']['bisec_Root'].mean()
    at_mean.append(temp)
    temp = df[df.moneyness == 'inTheMoney']['bisec_Root'].mean()
    in_mean.append(temp)
    temp = df[df.moneyness == 'outTheMoney']['bisec_Root'].mean()
    out_mean.append(temp)
amzn_mean = pd.DataFrame({'expiry': expiry_date, 'at-the-money': at_mean,
                         'out-of-the-money': out_mean, 'in-the-money': in_mean})
# spy
expiry_date = []
in_mean = []
out_mean = []
at_mean = []
for df in DATA1[1]:
    expiry_date.append(df.expiry.iloc[0])
    temp = df[df.moneyness == 'atTheMoney']['bisec_Root'].mean()
    at_mean.append(temp)
    temp = df[df.moneyness == 'inTheMoney']['bisec_Root'].mean()
    in_mean.append(temp)
    temp = df[df.moneyness == 'outTheMoney']['bisec_Root'].mean()
    out_mean.append(temp)
spy_mean = pd.DataFrame({'expiry': expiry_date, 'at-the-money': at_mean,
                         'out-of-the-money': out_mean, 'in-the-money': in_mean})


if __name__ == "__main__":
    # average vol
    print(amzn_mean)
    print(spy_mean)

    # compare results calculated by two methods
    info = ['date', 'expiry', 'contractSymbol', 'strike', 'optionPrice',
            'delta_t', 'bisec_Root', 'newton_Root', 'moneyness']
    temp = DATA1[0][5]
    temp[temp['moneyness'] == 'atTheMoney'][info]

    # running time
    print(pd.DataFrame({'method': ['bisection method', "newton's method"],
             'running time': [bisec_time, newton_time]}))

    # the average volatilities for every maturity, type, stock/ETF:
        # AMZN
    expiry_date = []
    call_vol = []
    put_vol = []
    for df in DATA1[0]:
        expiry_date.append(df.expiry.iloc[0])
        temp = df[df.type == 'c']['bisec_Root'].mean()
        call_vol.append(temp)
        temp = df[df.type == 'p']['bisec_Root'].mean()
        put_vol.append(temp)
    amzn_vol = pd.DataFrame({'expiry': expiry_date, 'vol of call': call_vol,
                             'vol of put': put_vol})
        # spy
    expiry_date = []
    call_vol = []
    put_vol = []
    for df in DATA1[1]:
        expiry_date.append(df.expiry.iloc[0])
        temp = df[df.type == 'c']['bisec_Root'].mean()
        call_vol.append(temp)
        temp = df[df.type == 'p']['bisec_Root'].mean()
        put_vol.append(temp)
    spy_vol = pd.DataFrame({'expiry': expiry_date, 'vol of call': call_vol,
                            'vol of put': put_vol})

        # concat above tables
    vol_table = pd.concat([amzn_vol, spy_vol], keys=['AMZN', 'SPY'])
    print(vol_table)


