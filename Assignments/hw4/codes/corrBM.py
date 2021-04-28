import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def vallina_payoff(s, k, Type):
    if Type == 'c':
        return np.maximum(s - k, 0)
    else:
        return np.maximum(k - s, 0)


def cholesky(A):
    L = np.zeros(A.shape)
    for i, (Ai, Li) in enumerate(zip(A, L)):
        for j, Lj in enumerate(L[:i + 1]):
            s = sum(Li[k] * Lj[k] for k in range(j))
            Li[j] = np.sqrt(Ai[i] - s) if (i == j) else \
                (1.0 / Lj[j] * (Ai[j] - s))
    return L


def multi_path(S, T, mu, q, sigma, n, m, A):
    # precomputes
    num_asset = A.shape[0]
    dt = T / n
    nudt = (mu - q - sigma ** 2 / 2) * dt
    sigdt = sigma * np.sqrt(dt)
    nudt = np.tile(nudt, (m, 1))
    sigdt = np.tile(sigdt, (m, 1))

    # simulation
    s0 = np.tile(np.log(S), (m, 1))
    lns = np.zeros((n + 1, m, num_asset))
    lns[0] = s0
    for i in range(n):
        # generate correlated guassians
        L = cholesky(A)
        guassians = np.random.normal(0, 1, size=(num_asset, m))
        corr_guassians = np.dot(L, guassians).transpose()
        # update paths
        lns[i + 1] = lns[i] + nudt + sigdt * (corr_guassians)
    s = np.exp(lns)
    return s


def vallina_basket_mc(S, K, T, mu, q, sigma, Type, n, m, A, alpha, r):
    # precomputes
    num_asset = A.shape[0]
    dt = T / n
    nudt = (mu - q - sigma ** 2 / 2) * dt
    sigdt = sigma * np.sqrt(dt)
    nudt = np.tile(nudt, (m, 1))
    sigdt = np.tile(sigdt, (m, 1))

    # simulation
    lns = np.tile(np.log(S), (m, 1))
    # s = np.zeros((m, num_asset))
    for i in range(n):
        # generate correlated guassians
        L = cholesky(A)
        guassians = np.random.normal(0, 1, size=(num_asset, m))
        corr_guassians = np.dot(L, guassians).transpose()
        lns = lns + nudt + sigdt * corr_guassians
    s = np.exp(lns)
    u = np.dot(s, alpha)
    payoffs = vallina_payoff(u, K, Type)
    p = payoffs.mean() * np.exp(-r * T)
    return p


def exotic_basket_mc(S, K, T, mu, q, sigma, n, m, A, alpha, r, B):
    s = multi_path(S, T, mu, q, sigma, n, m, A)
    payoff = 0

    s1 = s[:, :, 0]
    s2 = s[:, :, 1]
    s3 = s[:, :, 2]
    for i in range(m):
        s1_t = s1[:, i][-1]
        s2_t = s2[:, i][-1]
        s3_t = s3[:, i][-1]
        s2_max = np.max(s2[1:, i])
        s3_max = np.max(s3[1:, i])
        A2 = s2[1:, i].mean()
        A3 = s3[1:, i].mean()
        if s2_max > B:
            payoff += vallina_payoff(s2_t, K, 'c')
        elif s2_max > s3_max:
            payoff += np.max(s2_t ** 2 - K, 0)
        elif A2 > A3:
            payoff += vallina_payoff(A2, K, 'c')
        else:
            s_t = np.array([s1_t, s2_t, s3_t])
            u = np.dot(s_t, alpha)
            payoff += vallina_payoff(u, K, 'c')
    p = payoff / m * np.exp(-r * T)
    return p


if __name__ == '__main__':
    S = np.array([100, 101, 98])
    mu = np.array([0.03, 0.06, 0.02])
    q = 0
    sigma = np.array([0.05, 0.2, 0.15])
    A = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, -0.4], [0.2, -0.4, 1.0]])
    T = 100 / 365
    n = 100
    m = 100000
    K = 100
    r = 0.1
    B = 104
    alpha = np.ones(3) / 3

    # 3-a
    L = Matrix(cholesky(A))
    print(latex(L))

    # # 3-b
    # s = multi_path(S, T, mu, q, sigma, n, m, A)
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    #
    # s_0 = s[:, 0, :]
    # xline = np.arange(0, m)
    # yline = np.arange(0, n + 1)
    # for i in range(m):
    #     path0 = s[:, i, 0]
    #     path1 = s[:, i, 1]
    #     path2 = s[:, i, 2]
    #     ax.plot(np.array([i] * (n + 1)), yline, path0, 'red')
    #     ax.plot(np.array([i] * (n + 1)), yline, path1, 'green')
    #     ax.plot(np.array([i] * (n + 1)), yline, path2, 'orange')
    # ax.set_xlim(0,16)
    # ax.set_xlabel('path')
    # ax.set_ylabel('time step')
    # ax.set_zlabel('stock price')
    # plt.legend(['asset1','asset2', 'asset3'])
    # plt.show()

    # # 3-c
    # print(vallina_basket_mc(S, K, T, mu, q, sigma, 'c', n, m, A, alpha, 0.1))
    # print(vallina_basket_mc(S, K, T, mu, q, sigma, 'p', n, m, A, alpha, 0.1))

    # # 3-d
    # print(exotic_basket_mc(S, K, T, mu, q, sigma, n, m, A, alpha, r, B))
