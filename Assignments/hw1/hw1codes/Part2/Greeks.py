from BS_formula import *
import numpy as np
import pandas as pd
from impliedVol import DATA1

r = 0.07/100
# approximation functions
def prime(f, x, h=1e-4):
    return (f(x + h) - f(x)) / h


def dprime(f, x, h=1e-4):
    return (prime(f, x + h) - prime(f, x)) / h


def delta_approx(S, K, T, r, sigma, h=1e-4):
    s = lambda x: BS_formula('c', x, K, T, sigma, r)
    return prime(s, S, h)


def gamma_approx(S, K, T, r, sigma, h=1e-4):
    s = lambda x: BS_formula('c', x, K, T, sigma, r)
    return dprime(s, S, h)


def vega_approx(S, K, T, r, sigma, h=1e-4):
    sig = lambda x: BS_formula('c', S, K, T, x, r)
    return prime(sig, sigma, h)


for i in range(2):
    for df in DATA1[i]:
        df['delta'] = df.apply(lambda x:
                               delta(x.type, x.spotPrice, x.strike,
                                              x.delta_t, r, x.bisec_Root),
                              axis=1)
        df['delta_approx'] = df.apply(lambda x:
                                      delta_approx(x.spotPrice, x.strike,
                                                   x.delta_t, r, x.bisec_Root),
                                     axis=1)
        df['gamma'] = df.apply(lambda x:
                               gamma(x.spotPrice, x.strike, x.delta_t,
                                              r, x.bisec_Root),
                              axis=1)
        df['gamma_approx'] = df.apply(lambda x:
                                     gamma_approx(x.spotPrice, x.strike,
                                                 x.delta_t, r, x.bisec_Root),
                                     axis=1)
        df['vega'] = df.apply(lambda x:
                             vega(x.spotPrice, x.strike, x.delta_t,
                                 x.bisec_Root, r), axis=1)
        df['vega_approx'] = df.apply(lambda x:
                                    vega_approx(x.spotPrice, x.strike, x.delta_t,
                                               r, x.bisec_Root), axis=1)

greeks_info = ['contractSymbol', 'delta', 'delta_approx',
              'gamma', 'gamma_approx', 'vega', 'vega_approx']
temp = DATA1[1][9]
temp = temp[temp.type=='c']
print(temp)