import numpy as np
from scipy.stats import norm


def BS_formula(Type, S, K, T, sigma, r):
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if Type == 'c':
        return norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(-r * T)
    elif Type == 'p':
        return K * np.exp(-r * T) * norm.cdf(-d2) - norm.cdf(-d1) * S
    else:
        raise TypeError("Type must be 'c' for call, 'p' for put")


def vega(S, K, T, sigma, r):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return np.sqrt(T) * S * norm.pdf(d1)


def newton_method(f, f_prime, x0, tol=1e-6, N=100):
    for i in range(N):
        x1 = x0 - f(x0) / f_prime(x0)
        if abs(x1 - x0) < tol:
            break
        x0 = x1
    return x1


def bisection(f, a, b, tol=1e-6):
    if f(a) == 0:
        return a
    elif f(b) == 0:
        return b
    while abs(a - b) >= tol:
        c = (a + b) / 2
        if f(c) == 0:
            break
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return c


def get_impliedVol(Type, S, K, T, r, P):
    def price_diff(sigma):
        return BS_formula(Type, S, K, T, sigma, r) - P
    return bisection(price_diff, 0.001, 1)


