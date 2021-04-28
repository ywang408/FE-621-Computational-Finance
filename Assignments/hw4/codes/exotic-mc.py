import numpy as np


def vallina_payoff(s, k, Type):
    if Type == 'c':
        return np.maximum(s - k, 0)
    else:
        return np.maximum(k - s, 0)


def asian_mc(S, K, T, r, q, sigma, Type, n, m, look_at_times):
    # precomputes
    dt = T / n
    nudt = (r - q - sigma ** 2 / 2) * dt
    sigdt = sigma * np.sqrt(dt)
    s = []

    lns = np.ones(m) * np.log(S)
    for i in range(1, n + 1):
        guassians = np.random.normal(0, 1, size=m)
        lns += nudt + sigdt * guassians
        if (i / n * look_at_times) % 1 == 0:
            s.append(lns.copy())
    s = np.array(s)
    s = np.exp(s)
    s = np.mean(s, axis=0)
    payoffs = vallina_payoff(s, K, Type)
    p = payoffs.mean() * np.exp(-r * T)
    return p


def up_out_mc(S, K, T, r, q, sigma, Type, n, m, look_at_times, H):
    # precomputes
    dt = T / n
    nudt = (r - q - sigma ** 2 / 2) * dt
    sigdt = sigma * np.sqrt(dt)
    s = []

    lns = np.ones(m) * np.log(S)
    for i in range(1, n + 1):
        guassians = np.random.normal(0, 1, size=m)
        lns += nudt + sigdt * guassians
        if (i / n * look_at_times) % 1 == 0:
            s.append(lns.copy())
    s = np.array(s)
    s = np.exp(s)

    def check(path):
        if np.max(path) > H:
            return 0.0
        else:
            return path[-1]
    s = np.apply_along_axis(check, 0, s)
    payoffs = vallina_payoff(s, K, Type)
    p = payoffs.mean() * np.exp(-r * T)
    return p


if __name__ == '__main__':
    S = 100
    K = 100
    r = 0.03
    q = 0.01
    sigma = 0.2
    T = 1
    n = 120
    m = 1000000
    look_at_times = 12
    # print(asian_mc(S, K, T, r, q, sigma, 'c', n, m, look_at_times))
    print(up_out_mc(S, K, T, r, q, sigma, 'c', n, m, look_at_times, 110))
