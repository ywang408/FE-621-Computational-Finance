import numpy as np
import matplotlib.pyplot as plt


def f(x):
    if x == 0:
        return 1
    else:
        return np.sin(x) / x


def trapezoidal(f, a, b, N=1000000):
    x = np.linspace(a, b, N + 1)
    g = np.vectorize(f)
    return (b - a) / N * (np.sum(g(x)) - f(a) / 2 - f(b) / 2)


def simpson(f, a, b, N=1000000):
    x = np.linspace(a, b, N + 1)
    g = np.vectorize(f)
    h = (b - a) / N
    return h / 3 * (g(a) + np.sum(g(x[1:-1:2])) * 4
                    + np.sum(g(x[2:-1:2])) * 2 + g(b))


print(trapezoidal(f, -1e6, 1e6, 1000000))
print(simpson(f, -1e6, 1e6, 1000000))


def trap_trunc(a=1e5, N=100000):
    return abs(trapezoidal(f, -a, a, N) - np.pi)


def simp_trunc(a=1e5, N=100000):
    return abs(simpson(f, -a, a, N) - np.pi)


def diff(a=1e5, N=100000):
    return trapezoidal(f, -a, a, N) - simpson(f, -a, a, N)


# a is fixed, N increases
N = np.arange(50, 10000, 50)
error_t1 = [trap_trunc(N=i) for i in N]
error_s1 = [simp_trunc(N=i) for i in N]
diff1 = [diff(N=i) for i in N]

# N is fixed, a increases
a = np.arange(50, 10000, 50)
error_t2 = [trap_trunc(a=i) for i in a]
error_s2 = [simp_trunc(a=i) for i in a]
diff2 = [diff(a=i) for i in a]

# error plots
plt.figure(1)
t1 = plt.subplot(2, 2, 1)
plt.plot(N,error_t1)
plt.title("a fixed, trapezoidal's error")
plt.ylabel('error')
t2 = plt.subplot(2, 2, 2)
plt.plot(a, error_t2)
plt.title("N fixed, trapezoidal's error")
plt.ylabel('error')

s1 = plt.subplot(2, 2, 3)
plt.plot(N, error_s1)
plt.title("a fixed, simpson's error", y=-0.35)
plt.ylabel('error')
s2 = plt.subplot(2, 2, 4)
plt.plot(N, error_s2)
plt.title("N fixed, simpson's error", y=-0.35)
plt.ylabel('error')
plt.show()

# diff plot
plt.figure(2)
t1 = plt.subplot(2, 1, 1)
plt.plot(N, diff1)
plt.title('diff when a fixed')
t2 = plt.subplot(2, 1, 2)
plt.plot(a, diff2)
plt.title('diff when N fixed', y=-0.3)
plt.show()


def integral(f, a, b, tol=1e-4, rule=trapezoidal):
    step = 10
    I_k = rule(f, a, b, step)
    I_k1 = rule(f, a, b, step+1)
    while abs(I_k - I_k1) >= tol:
        step += 1
        I_k = I_k1
        I_k1 = rule(f, a, b, step+1)
    return I_k1, step


# 3
res1, step1 = integral(f, -1e4, 1e4)
res2, step2 = integral(f, -1e4, 1e4, rule=simpson)
print("The value using trapezoidal is {0}, and take {1} steps".format(res1, step1))
print("The value using simpson is {0}, and take {1} steps".format(res2, step2))


g = lambda x: 1 + np.exp(-x**2)*np.sin(8*x**(2/3))
res3, step3 = integral(g, 0, 2)
res4, step4 = integral(g, 0, 2, rule=simpson)
# 4
print("The value using trapezoidal is {0}, and take {1} steps".format(res3, step3))
print("The value using simpson is {0}, and take {1} steps".format(res4, step4))