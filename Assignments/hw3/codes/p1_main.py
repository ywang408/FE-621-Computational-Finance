import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from finite_diff_methods import *

# params
S = 100
K = 100
T = 1
sigma = 0.2
r = 0.06
q = 0.02

epsilon = 0.0001
dt = epsilon / (3 * sigma ** 2 + 1)
dx = sigma * np.sqrt(3 * dt)
N = int(np.ceil(T / dt))
Nj = int(np.ceil((2 * np.sqrt(3 * N) - 1) / 2))


# # part e
# print("result of part e:")
# print("dt: ", dt)
# print("dx: ", dx)
# print("N: ", N)
# print("Nj: ", Nj)
#
# ec = e_fdm(S, K, T, r, sigma, q, N, Nj, dx, 'c', 'e')
# ep = e_fdm(S, K, T, r, sigma, q, N, Nj, dx, 'p', 'e')
# print("call of explicit method is: {0}, "
#       "put of explicit method is: {1}".format(ec, ep))
# ic = i_fdm(S, K, T, r, sigma, q, N, Nj, dx, 'c', 'e')
# ip = i_fdm(S, K, T, r, sigma, q, N, Nj, dx, 'p', 'e')
# print("call of implicit method is: {0}, "
#       "put of implicit method is: {1}".format(ic, ip))
# cc = cn_fdm(S, K, T, r, sigma, q, N, Nj, dx, 'c', 'e')
# cp = cn_fdm(S, K, T, r, sigma, q, N, Nj, dx, 'p', 'e')
# print("call of Crank-Nicolson method is: {0}, "
#       "put of Crank-Nicolson method is: {1}".format(cc, cp))


# part f
def BS_formula(Type, S, K, T, sigma, r):
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if Type == 'c':
        return norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(-r * T)
    elif Type == 'p':
        return K * np.exp(-r * T) * norm.cdf(-d2) - norm.cdf(-d1) * S
    else:
        raise TypeError("Type must be 'c' for call, 'p' for put")


def get_iter(S, K, T, r, sigma, q, op_type, method):
    N = 50
    dt = T / N
    dx = sigma * np.sqrt(3 * dt)
    Nj = int(np.ceil((2 * np.sqrt(3 * N) - 1) / 2))
    fd_price = method(S, K, T, r, sigma, q, N, Nj, dx, op_type, 'e')
    bs_price = BS_formula(op_type, S, K, T, sigma, r, q)
    iter = 0
    while abs(fd_price - bs_price) > epsilon:
        N += 100
        dt = T / N
        dx = sigma * np.sqrt(3 * dt)
        Nj = int(np.ceil((2 * np.sqrt(3 * N) - 1) / 2))
        fd_price = method(S, K, T, r, sigma, q, N, Nj, dx, op_type, 'e')
        bs_price = BS_formula(op_type, S, K, T, sigma, r, q)
        iter += 1
    return iter


# # esc = get_iter(S, K, T, r, sigma, q, 'c', e_fdm)
# # esp = get_iter(S, K, T, r, sigma, q, 'p', e_fdm)
# print("step1 of explicit method is: {0}, "
#       "step2 of explicit method is: {1}".format(76, 154))
# # isc = get_iter(S, K, T, r, sigma, q, 'c', i_fdm)
# # isp = get_iter(S, K, T, r, sigma, q, 'p', i_fdm)
# print("step1 of explicit method is: {0}, "
#       "step2 of explicit method is: {1}".format(193, 321))
# # csc = get_iter(S, K, T, r, sigma, q, 'c', cn_fdm)
# # csp = get_iter(S, K, T, r, sigma, q, 'p', cn_fdm)
# print("step1 of Crank-Nicolson method is: {0}, "
#       "step2 of Crank-Nicolson method is: {1}".format(139, 237))

# # part g
#
# def prob(T, r, sigma, q, N, dx):
#     dt = T / N
#     nu = r - q - sigma ** 2 / 2
#     pu = - 0.5 * dt * ((sigma / dx) ** 2 + nu / dx)
#     pm = 1 + dt * (sigma / dx) ** 2 + r * dt
#     pd = - 0.5 * dt * ((sigma / dx) ** 2 - nu / dx)
#     return pu, pm, pd
#
#
# sig = np.arange(0.05, 0.61, 0.05)
# pu, pm, pd = prob(T, r, sig, q, N, dx)
# plt.figure(1)
# plt.xlabel("sigma")
# plt.ylabel("probs")
# plt.title("probs of implicit finite difference method")
# plt.plot(sig, pu, label='pu')
# plt.plot(sig, pm, label='pm')
# # plt.plot(sig, pd, label = 'pd')
# plt.legend()
# plt.show()

# part i
delta, gamma, theta = delta_gamma_theta(
    S, K, T, r, sigma, q, N, Nj, dx, 'c')
vega = vega(S, K, T, r, sigma, q, N, Nj, dx, 'c')

print("delta: ", delta)
print("gamma: ", gamma)
print("vega: ", vega)
print("theta: ", theta)
