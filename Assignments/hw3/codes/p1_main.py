import numpy as np
import matplotlib.pyplot as plt
from finite_diff_methods import *
from finite_diff_iters import *

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
        print(fd_price, bs_price)
    return iter


esc = get_iter(S, K, T, r, sigma, q, 'c', e_fdm)
esp = get_iter(S, K, T, r, sigma, q, 'p', e_fdm)
print("step1 of explicit method is: {0}, "
      "step2 of explicit method is: {1}".format(esc, esp))
isc = get_iter(S, K, T, r, sigma, q, 'c', i_fdm)
isp = get_iter(S, K, T, r, sigma, q, 'p', i_fdm)
print("step1 of explicit method is: {0}, "
      "step2 of explicit method is: {1}".format(isc, isp))
csc = get_iter(S, K, T, r, sigma, q, 'c', cn_fdm)
csp = get_iter(S, K, T, r, sigma, q, 'p', cn_fdm)
print("step1 of Crank-Nicolson method is: {0}, "
      "step2 of Crank-Nicolson method is: {1}".format(csc, csp))