import numpy as np


def payoff(op_type, s, k):
    if op_type == 'c':
        return np.maximum(s - k, 0)
    elif op_type == 'p':
        return np.maximum(k - s, 0)
    else:
        raise ValueError("undefined option type")


def e_fdm(S, K, T, r, sigma, q, N, Nj, dx, op_type, style):
    # precompute
    dt = T / N
    nu = r - q - sigma ** 2 / 2
    pu = 0.5 * dt * ((sigma / dx) ** 2 + nu / dx)
    pm = 1 - dt * (sigma / dx) ** 2 - r * dt
    pd = 0.5 * dt * ((sigma / dx) ** 2 - nu / dx)

    # stock price and payoff at maturity
    st = np.arange(Nj, -Nj - 1, -1)
    st = np.exp(st * dx) * S
    p = payoff(op_type, st, K)

    def backward(p):
        temp1 = np.roll(p, -1)
        temp2 = np.roll(p, -2)
        temp3 = p * pu + temp1 * pm + temp2 * pd
        p[1:-1] = temp3[0:-2]
        if op_type == 'c':
            p[0] = p[1] + (st[0] - st[1])
            p[-1] = p[-2]
        elif op_type == 'p':
            p[0] = p[1]
            p[-1] = p[-2] + (st[-2] - st[-1])
        if style == 'a':
            p = np.maximum(p, payoff(op_type, st, K))

    for i in range(N):
        backward(p)

    return p[N]


def i_fdm(S, K, T, r, sigma, q, N, Nj, dx, op_type, style):
    # precompute constants
    dt = T / N
    nu = r - q - sigma ** 2 / 2
    pu = - 0.5 * dt * ((sigma / dx) ** 2 + nu / dx)
    pm = 1 + dt * (sigma / dx) ** 2 + r * dt
    pd = - 0.5 * dt * ((sigma / dx) ** 2 - nu / dx)

    # construct tridiagnal matrix
    l1 = np.zeros((1, 2 * Nj + 1))
    l2 = np.zeros((1, 2 * Nj + 1))
    l1[0][0] = 1
    l1[0][1] = -1
    l2[0][-1] = -1
    l2[0][-2] = 1
    rows = 2 * Nj - 1
    cols = 2 * Nj + 1
    a = np.eye(rows, cols, 0) * pu \
        + np.eye(rows, cols, 1) * pm \
        + np.eye(rows, cols, 2) * pd
    a = np.r_[l1, a, l2]

    # stock price and payoff at maturity
    st = np.arange(Nj, -Nj - 1, -1)
    st = np.exp(st * dx) * S
    p = payoff(op_type, st, K)

    # lambda
    if op_type == 'c':
        lambda_u = st[0] - st[1]
        lambda_l = 0
    elif op_type == 'p':
        lambda_u = 0
        lambda_l = st[-2] - st[-1]

    # backward calculation
    def backward(p):
        b = np.append(lambda_u, p[1:-1])
        b = np.append(b, lambda_l)
        x = np.linalg.solve(a, b)
        print(x.shape)
        return x

    for i in range(N):
        p = backward(p)

    return p[N]


def cn_fdm(S, K, T, r, sigma, q, N, Nj, dx, op_type, style):
    # precompute constants
    dt = T / N
    nu = r - q - sigma ** 2 / 2
    pu = - 0.25 * dt * ((sigma / dx) ** 2 + nu / dx)
    pm = 1 + 0.5 * dt * (sigma / dx) ** 2 + 0.5 * r * dt
    pd = - 0.25 * dt * ((sigma / dx) ** 2 - nu / dx)

    # construct tridiagnal matrix
    l1 = np.zeros((1, 2 * Nj + 1))
    l2 = np.zeros((1, 2 * Nj + 1))
    l1[0][0] = 1
    l1[0][1] = -1
    l2[0][-1] = -1
    l2[0][-2] = 1
    rows = 2 * Nj - 1
    cols = 2 * Nj + 1
    a = np.eye(rows, cols, 0) * pu \
        + np.eye(rows, cols, 1) * pm \
        + np.eye(rows, cols, 2) * pd
    a = np.r_[l1, a, l2]

    # stock price and payoff at maturity
    st = np.arange(Nj, -Nj - 1, -1)
    st = np.exp(st * dx) * S
    p = payoff(op_type, st, K)

    # lambda
    if op_type == 'c':
        lambda_u = st[0] - st[1]
        lambda_l = 0
    elif op_type == 'p':
        lambda_u = 0
        lambda_l = st[-2] - st[-1]

    # backward calculation
    def backward(p):
        temp1 = np.roll(p, -1)
        temp2 = np.roll(p, -2)
        temp3 = -p * pu - temp1 * (pm-2) - temp2 * pd
        p[1:-1] = temp3[0:-2]
        b = np.append(lambda_u, p[1:-1])
        b = np.append(b, lambda_l)
        x = np.linalg.solve(a, b)
        return x

    for i in range(N):
        p = backward(p)

    return p[N]


if __name__ == "__main__":
    K = 40
    T = 0.5
    S = 42
    sigma = 0.2
    r = 0.1
    q = 0
    N = 300
    Nj = 300
    dx = 0.002
    a = cn_fdm(S, K, T, r, sigma, q, N, Nj, dx, 'c', 'e')
    print(a)
