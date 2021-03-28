import numpy as np


def binomial_tree(S, K, T, r, sigma, N, Type, style):
    """
    Binomial tree method for option pricing
    :param S: spot price
    :param K: strike
    :param T: time to maturity
    :param r: risk-free interest
    :param sigma: volatility of underlying asset
    :param N: steps of tree
    :param Type: option type, 'c' or 'p'
    :param style: option style, 'a' or 'e'
    :return: price of option interested in
    """
    # check illegal Type and style
    if Type not in ['c', 'p']:
        raise TypeError("option type should be 'c' for call,"
                        "'p' for put")
    if style not in ['e', 'a']:
        raise TypeError("option style should be 'a' for American option,"
                        "'e' for European option")
    # delta_t, delta_x, prob of up and down
    t = T / N
    delta_x = np.sqrt((r - sigma ** 2 / 2) ** 2 * t ** 2 + sigma ** 2 * t)
    p_u = 0.5 + 0.5 * (r - sigma ** 2 / 2) * t / delta_x
    p_d = 1 - p_u
    print(delta_x)
    print(p_u, p_d)
    # construct stock price tree
    def tree_construction():
        s = [np.array([S])]
        for i in range(N):
            temp = np.exp(delta_x) * s[i]
            temp = np.append(temp, s[i][-1] * np.exp(-delta_x))
            s.append(temp)
        return s

    # payoff
    def payoff(s):
        if Type == 'c':
            return np.maximum(s - K, 0)
        elif Type == 'p':
            return np.maximum(K - s, 0)
        else:
            raise TypeError("Illegal option type!")

    # backward calculation
    def backward(s, p):
        temp = np.roll(p, -1)
        temp = p * p_u + temp * p_d
        temp = temp * np.exp(-r * t)
        temp = np.delete(temp, -1)
        if style == 'e':
            return temp
        elif style == 'a':
            current_p = payoff(s)
            temp = np.maximum(temp, current_p)
            return temp

    s = tree_construction()
    print("s:")
    print(s)
    # nodes at maturity
    s_t = s[-1]
    p = payoff(s_t)
    print("p:")
    print(p)
    while len(p) > 1:
        p = backward(s[len(p) - 2], p)
        print("p:")
        print(p)
    return float(p)


def trinomial_tree(S, K, barrier, T, r, sigma, N, Type, style):
    # check illegal Type and style
    if Type not in ['c', 'p']:
        raise TypeError("option type should be 'c' for call,"
                        "'p' for put")
    if style not in ['e', 'a']:
        raise TypeError("option style should be 'a' for American option,"
                        "'e' for European option")
    # initialization
    t = T / N
    D = r - sigma ** 2 / 2
    delta_x = sigma * np.sqrt(3 * t)
    p_u = 0.5 * ((sigma ** 2 * t + D ** 2 * t ** 2) / delta_x ** 2 + D * t / delta_x)
    p_m = 1 - (sigma ** 2 * t + D ** 2 * t ** 2) / delta_x ** 2
    p_d = 0.5 * ((sigma ** 2 * t + D ** 2 * t ** 2) / delta_x ** 2 - D * t / delta_x)
    # print(delta_x)
    # u = np.exp(delta_x)
    # print(u, 1/u)
    # print(p_u, p_m, p_d)
    def tree_construction():
        s = [np.array([S])]
        for i in range(N):
            temp = np.exp(delta_x) * s[i]
            temp = np.append(temp, s[i][-1])
            temp = np.append(temp, s[i][-1] * np.exp(-delta_x))
            s.append(temp)
        return s

    # payoff
    def payoff(s):
        if Type == 'c':
            return np.maximum(s - K, 0)
        elif Type == 'p':
            return np.maximum(K - s, 0)
        else:
            raise TypeError("Illegal option type!")

    def backward(s, p):
        temp1 = np.roll(p, -1)
        temp2 = np.roll(p, -2)
        temp = p * p_u + temp1 * p_m + temp2 * p_d
        temp = temp * np.exp(-r * t)
        temp = np.delete(temp, -1)
        temp = np.delete(temp, -1)
        if style == 'e':
            return temp
        elif style == 'a':
            current_p = payoff(s)
            temp = np.maximum(temp, current_p)
            return temp

    def backward_barrier(s, p):
        temp1 = np.roll(p, -1)
        temp2 = np.roll(p, -2)
        temp = p * p_u + temp1 * p_m + temp2 * p_d
        temp = temp * np.exp(-r * t)
        temp = np.delete(temp, -1)
        temp = np.delete(temp, -1)
        if barrier[1] == 'D':
            temp[s <= barrier[0]] = 0
            if style == 'e':
                return temp
            elif style == 'a':
                current_p = payoff(s)
                temp = np.maximum(temp, current_p)
                temp[s <= barrier[0]] = 0
                return temp
        elif barrier[1] == 'U':
            temp[s >= barrier[0]] = 0
            if style == 'e':
                return temp
            elif style == 'a':
                current_p = payoff(s)
                temp = np.maximum(temp, current_p)
                temp[s >= barrier[0]] = 0
                return temp

    s = tree_construction()
    s_t = s[-1]
    # print('s')
    # print(s)
    p = payoff(s_t)
    # without barrier
    if barrier == 0:
        p = payoff(s_t)
        # print("p")
        # print(p)
        while len(p) > 1:
            p = backward(s[len(p) // 2 - 1], p)
            # print(p)
        return float(p)

    # with barrier
    if barrier[1] == 'D':
        p[s_t <= barrier[0]] = 0
    elif barrier[1] == 'U':
        p[s_t >= barrier[0]] = 0
    if barrier[2] == 'O':
        while len(p) > 1:
            p = backward_barrier(s[len(p) // 2 - 1], p)
        return float(p)
    elif barrier[2] == 'I':
        out = trinomial_tree(S, K, [barrier[0], barrier[1], 'O'], T, r, sigma, N, Type, style)
        v = trinomial_tree(S, K, 0, T, r, sigma, N, Type, style)
        return v - out


if __name__ == '__main__':
    S = 50
    K = 49
    r = 0.05
    sigma = 0.3
    T = 0.75
    N = 3
    # print(binomial_tree(S, K, T, r, sigma, N, 'p', 'e'))
    print(trinomial_tree(S, K, 0, T, r, sigma, N, 'p', 'a'))
    # a = np.exp(0.05*0.25)
    # u = np.exp(sigma * np.sqrt(0.25))
    # d = 1/u
    # print(u, d)
    # print((a-d)/(u-d))