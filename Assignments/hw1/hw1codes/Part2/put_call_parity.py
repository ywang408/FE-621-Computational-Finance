import numpy as np
import pandas as pd
from impliedVol import DATA1


def parity(Type, S, K, T, r, P):
    if Type == 'c':
        return P - S + K * np.exp(-r * T)
    elif Type == 'p':
        return P + S - K * np.exp(-r * T)


r = 0.07/100
for i in range(2):
    for df in DATA1[i]:
        df['parity_price'] = df.apply(lambda x: parity(x.type, x.spotPrice,
                                                       x.strike, x.delta_t, r, x.optionPrice),
                                      axis=1)

# choose data with the same strike
temp = DATA1[1][9]
dup = temp['strike'].duplicated(keep=False)
temp = temp[dup]
temp = temp.sort_values(by='strike')
info = ['strike', 'bid', 'ask', 'parity_price']

# merge call and put dataframe
temp_p = temp[temp.type == 'p'][info]
temp_c = temp[temp.type == 'c'][info]
temp = pd.merge(temp_p, temp_c, on='strike')
temp = temp.rename(columns={'bid_x': 'bid of put', 'ask_x': 'ask of put', 'parity_price_x': 'parity price by put',
                            'bid_y': 'bid of call', 'ask_y': 'ask of call', 'parity_price_y': 'parity price by call'})

print(temp)

# calculate average diff
bp_diff = (temp['bid of put'] - temp['parity price by call']).mean()
ap_diff = (temp['ask of put'] - temp['parity price by call']).mean()
bc_diff = (temp['bid of call'] - temp['parity price by put']).mean()
ac_diff = (temp['bid of call'] - temp['parity price by put']).mean()
print(pd.DataFrame({'bid': [bp_diff, bc_diff],
                    'ask': [ap_diff, ac_diff]}, index=['put', 'call']))
