from scipy.special import comb, perm
from sympy import *
import numpy as np
from math import exp,sqrt

def L(x):
    sum = 0
    for k in range(N + 1):
        sum += (-1) ** k / perm(k, k) * comb(N, k) * x ** k
    return sum


def L_(x):
    sum = 0
    for k in range(N + 1):
        sum += (-1) ** k / perm(k, k) * comb(N, k) * k * x**(k-1)
    return sum


def weight(x_k):
    return exp(x_k) / x_k * (perm(N, N)/L_(x_k))**2


def f(x):
    return exp(-x**2/2)


N = 20
x = symbols('x')
x = solve(L(x), x)
x = np.array(x)
# w = weight(x)
w = np.array([])
for x_k in x:
    w = np.append(w, weight(x_k))
res = 0
for k in range(0,N):
    res += w[k] * f(x[k])
print(res)

