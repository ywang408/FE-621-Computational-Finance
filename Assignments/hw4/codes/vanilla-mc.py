import numpy as np
from scipy.stats import norm
import pandas as pd
import time


def bs_delta(S, K, T, r, q, sigma, Type):
    nu = r - q + sigma ** 2 / 2
    d1 = (np.log(S / K) + nu * T) / (sigma * np.sqrt(T))
    if Type == 'c':
        return norm.cdf(d1)
    if Type == 'p':
        return norm.cdf(d1) - 1


def vanilla_payoff(s, k, Type):
    if Type == 'c':
        return np.maximum(s - k, 0)
    else:
        return np.maximum(k - s, 0)


def vanilla_mc(S, K, T, r, q, sigma, Type, n, m, reduce):
    # reduce: 0 - not apply variate reduction methods
    # 'a'-antithetic, 'c'-control variate, 'b'-both

    # precomputes
    dt = T / n
    nudt = (r - q - sigma ** 2 / 2) * dt
    sigdt = sigma * np.sqrt(dt)
    erddt = np.exp((r - q) * dt)
    beta = -1

    # no methods for reducing variance
    if reduce == 0:
        lns = np.ones(m) * np.log(S)
        for i in range(n):
            guassians = np.random.normal(0, 1, size=m)
            lns += nudt + sigdt * guassians
        s = np.exp(lns)
        payoffs = vanilla_payoff(s, K, Type)
        p = payoffs.mean() * np.exp(-r * T)
        std = payoffs.std() * np.exp(-r * T)
        return p, std

    # antithetic variates
    if reduce == 'a':
        lns1 = np.ones(m) * np.log(S)
        lns2 = np.ones(m) * np.log(S)
        for i in range(n):
            guassians = np.random.normal(0, 1, size=m)
            lns1 += nudt + sigdt * guassians
            lns2 += nudt + sigdt * (-guassians)
        s1 = np.exp(lns1)
        s2 = np.exp(lns2)
        payoffs = (vanilla_payoff(s1, K, Type) + vanilla_payoff(s2, K, Type)) / 2
        p = payoffs.mean() * np.exp(-r * T)
        std = payoffs.std() * np.exp(-r * T)
        return p, std

    # control variate
    if reduce == 'c':
        s = np.ones(m) * S
        cv = 0
        for i in range(n):
            guassians = np.random.normal(0, 1, size=m)
            delta = bs_delta(s, K, T - i * dt, r, q, sigma, Type)
            s_n = s.copy() * np.exp(nudt + sigdt * guassians)
            cv += delta * (s_n - s * erddt)
            s = s_n
        payoffs = vanilla_payoff(s, K, Type) + beta * cv
        p = payoffs.mean() * np.exp(-r * T)
        std = payoffs.std() * np.exp(-r * T)
        return p, std

    # apply both methods
    if reduce == 'b':
        s1 = np.ones(m) * S
        s2 = np.ones(m) * S
        cv1 = 0
        cv2 = 0
        for i in range(n):
            guassians = np.random.normal(0, 1, size=m)
            delta1 = bs_delta(s1, K, T - i * dt, r, q, sigma, Type)
            delta2 = bs_delta(s2, K, T - i * dt, r, q, sigma, Type)
            s_n1 = s1.copy() * np.exp(nudt + sigdt * guassians)
            s_n2 = s2.copy() * np.exp(nudt - sigdt * guassians)
            cv1 += delta1 * (s_n1 - s1 * erddt)
            cv2 += delta2 * (s_n2 - s2 * erddt)
            s1 = s_n1
            s2 = s_n2
        payoffs = (vanilla_payoff(s1, K, Type) + vanilla_payoff(s2, K, Type)
                   + beta * cv1 + beta * cv2) / 2
        p = payoffs.mean() * np.exp(-r * T)
        std = payoffs.std() * np.exp(-r * T)
        return p, std


if __name__ == "__main__":
    S = 100
    K = 100
    r = 0.03
    q = 0.01
    sigma = 0.2
    T = 1

    # # part b
    # n = 300
    # m = 1000000
    # method = [0, 'a', 'c', 'b']
    # res = pd.DataFrame([])
    # for i, reduce in enumerate(method):
    #     start = time.time()
    #     p, std = vanilla_mc(S, K, T, r, q, sigma, 'c', n, m, reduce)
    #     t = time.time() - start
    #     res_tmp = pd.Series([p,std,t])
    #     res = res.append(res_tmp, ignore_index=True)
    # res.columns = ['price', 'std', 'time']
    # res.index = ['No method', 'Antithetic Method', 'Variate Control Method', 'Combined']
    # res = res.round(3)
    # res.to_csv("../attachments/1-b.csv")

    # part a
    n = [300, 500, 700]
    m = [1000000, 3000000, 5000000]
    price = pd.DataFrame([])
    std = pd.DataFrame([])
    t = pd.DataFrame([])
    for i in n:
        price_ = []
        std_ = []
        t_ = []
        for j in m:
            start = time.time()
            p_j, std_j = vanilla_mc(S, K, T, r, q, sigma, 'c', i, j, 0)
            t_j = time.time() - start
            price_.append(p_j)
            std_.append(std_j)
            t_.append(t_j)
        price = price.append(pd.Series(price_), ignore_index=True)
        std = std.append(pd.Series(std_), ignore_index=True)
        t = t.append(pd.Series(t_), ignore_index=True)
    data = [price, std, t]
    for i, df in enumerate(data):
        df.columns = m
        df.index = n
        df = df.round(3)
        path = "../attachments/1-a-" + str(i) + ".csv"
        df.to_csv(path)
