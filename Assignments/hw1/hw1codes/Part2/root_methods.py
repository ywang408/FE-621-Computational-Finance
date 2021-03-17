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


def newton_method(f, f_prime, x0, tol=1e-6, N=100):
    for i in range(N):
        x1 = x0 - f(x0)/f_prime(x0)
        if abs(x1- x0) < tol:
            break
        x0 = x1
    return x1


def test_f(x):
    return x**2 - 2


def test_f_prime(x):
    return 2*x


if __name__ == '__main__':
    print(bisection(test_f, 0, 1.5))
    print(newton_method(test_f, test_f_prime, -1))
