import numpy as np
from scipy.stats import norm


def bs_delta(S, K, T, r, q, sigma, Type):
    nu = r - q + sigma ** 2 / 2
    d1 = (np.log(S / K) + nu * T) / (sigma * np.sqrt(T))
    if Type == 'c':
        return norm.cdf(d1)
    if Type == 'p':
        return norm.cdf(d1) - 1


def vallina_payoff(s, k, Type):
    if Type == 'c':
        return np.maximum(s - k, 0)
    else:
        return np.maximum(k - s, 0)


def vallina_mc(S, K, T, r, q, sigma, Type, n, m, reduce):
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
        payoffs = vallina_payoff(s, K, Type)
        p = payoffs.mean() * np.exp(-r * T)
        return p

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
        payoffs = (vallina_payoff(s1, K, Type) + vallina_payoff(s2, K, Type)) / 2
        p = payoffs.mean() * np.exp(-r * T)
        return p

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
        payoffs = vallina_payoff(s, K, Type) + beta*cv
        p = payoffs.mean() * np.exp(-r * T)
        return p

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
            s_n2 = s2.copy() * np.exp(nudt + sigdt * guassians)
            cv1 += delta1 * (s_n1 - s1 * erddt)
            cv2 += delta2 * (s_n2 - s2 * erddt)
            s1 = s_n1
            s2 = s_n2
        payoffs = (vallina_payoff(s1, K, Type) + vallina_payoff(s2, K, Type)
                   + beta*cv1 + beta*cv2) / 2
        p = payoffs.mean() * np.exp(-r * T)
        return p


if __name__ == "__main__":
    print(vallina_mc(100, 100, 1, 0.06, 0.03, 0.2, 'c', 100, 1000000, 0))
    print(vallina_mc(100, 100, 1, 0.06, 0.03, 0.2, 'c', 100, 1000000, 'a'))
    print(vallina_mc(100, 100, 1, 0.06, 0.03, 0.2, 'c', 100, 1000000, 'c'))
    print(vallina_mc(100, 100, 1, 0.06, 0.03, 0.2, 'c', 100, 1000000, 'b'))
