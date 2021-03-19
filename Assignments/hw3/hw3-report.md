---
title: FE-621 Homework3
author: You Wang
date: 03/18/2021
---

## Problem 1

### (a)

For the explicit method the final discretized equation is:

$$
-\frac{V_{i+1, j}-V_{i, j}}{\Delta t}=\nu \frac{V_{i+1, j+1}-V_{i+1, j-1}}{2 \Delta x}+\frac{1}{2} \sigma^{2} \frac{V_{i+1, j+1}-2 V_{i+1, j}+V_{i+1, j-1}}{\Delta x^{2}}-r V_{i+1, j}
$$

where $\nu = r - q - \frac{\sigma^2}{2}$ 

Rearrange the equation we get: 

$$
\begin{aligned}
V_{i, j} &=p_{\mathrm{u}} V_{i+1 . j+1}+p_{\mathrm{m}} V_{i+1 . j}+p_{\mathrm{d}} V_{i+1, j-1} \\
p_{\mathrm{u}} &=\Delta t\left(\frac{\sigma^{2}}{2 \Delta x^{2}}+\frac{\nu}{2 \Delta x}\right) \\
p_{\mathrm{m}} &=1-\Delta t \frac{\sigma^{2}}{\Delta x^{2}}-r \Delta t \\
p_{\mathrm{d}} &=\Delta t\left(\frac{\sigma^{2}}{2 \Delta x^{2}}-\frac{\nu}{2 \Delta x}\right)
\end{aligned}
$$

At last, the boundary conditions:

$$
V_{N_j,j} = \begin{cases}
    V_{N_j-1, j+1} & for \ calls\\
    0 & for \ puts\\ 
\end{cases}
$$

$$
V_{-N_j,j} = \begin{cases}
    0 & for \ calls\\
    V_{-N_j+1, j+1} & for \ puts\\ 
\end{cases}
$$

Then we can implement explicit method in our program:

1. First of all, we define a `payoff` function to calculate payoffs:

    ```python
    def payoff(op_type, s, k):
        if op_type == 'c':
            return np.maximum(s - k, 0)
        elif op_type == 'p':
            return np.maximum(k - s, 0)
        else:
            raise ValueError("undefined option type")
    ```

2. Next define the explicit function as `e_fdm`:

    - In the function, we first precomputes constants $\Delta_t, \nu, p_u, p_m, p_d$.

    - Then use `np.arange` to generate a list from $N_j$ to $-N_j$ with step 1: 
  
    $$l = (N_j, N_j-1 \cdots, 0, \cdots -N_j+1, -N_j)$$

    $$S_t= S_0  e^{l \cdot \Delta_x}$$

    - Next we define a `backward` function to do backward calculation, and here use `np.roll` to do vectorized calculation instead of looping; Besides in the function we apply the boundary and early exercise (if it's an American option) condition.

    ![Basic idea of backward calculation](images/2021-03-18-18-57-06.png)

    - At last of `e_fdm` function, do `backward` for $N$ times and return the result.

```python
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

```

### (b)

$$
-\frac{V_{i+1, j}-V_{i, j}}{\Delta t}=\frac{1}{2} \sigma^{2} \frac{V_{i, j+1}-2 V_{i, j}+V_{i, j-1}}{\Delta x^{2}}+\nu \frac{V_{i, j+1}-V_{i, j-1}}{2 \Delta x}-r V_{i, j}
$$

It can be written as:

$$
\begin{aligned}
V_{i+1, i} &=p_{\mathrm{u}} V_{i, i+1}+p_{\mathrm{m}} V_{i, j}+p_{\mathrm{d}} V_{i, i-1} \\
p_{\mathrm{u}}&=-\frac{1}{2} \Delta t\left(\frac{\sigma^{2}}{\Delta x^{2}}+\frac{\nu}{\Delta x}\right) \\
p_{\mathrm{m}}&=1+\Delta t \frac{\sigma^{2}}{\Delta x^{2}}+r \Delta t \\
p_{d}&=-\frac{1}{2} \Delta t\left(\frac{\sigma^{2}}{\Delta x^{2}}-\frac{\nu}{\Delta x}\right)
\end{aligned}
$$

The boundary conditions are the same as we talked before in Problem [a](#a). 

So it can be expressed in the matrix form: $Ax=b$

$$
\left[\begin{array}{ccccccc}
1 & -1 & 0 & 0 & 0 & \ldots & 0 \\
p_u &p_m & p_d & 0 & 0 & \ldots & 0 \\
0 & p_u &p_m & p_d & 0 & \ldots & 0 \\
\vdots & \ddots & \ddots & \ddots & \ddots & \ddots & \vdots \\
0 & 0 & \ddots & \ddots & p_m & p_d & 0 \\
0 & 0 & 0 & \ddots & p_u &p_m & p_d \\
0 & 0 & 0 & \ldots & 0 & 1 & -1
\end{array}\right]\left[\begin{array}{c}
V_{i, N_{j}} \\
V_{i, N_{j}-1} \\
V_{i, N_{j}-2} \\
\vdots \\
\vdots \\
V_{i,-N_{j}+1} \\
V_{i,-N_{j}}
\end{array}\right]=\left[\begin{array}{c}
\lambda_{U} \\
V_{i+1, N_{j}-1} \\
V_{i+1, N_{j}-2} \\
\vdots \\
\vdots \\
V_{i+1,-N_{j}+1} \\
\lambda_{L}
\end{array}\right]
$$

To implement it in python:

1. First precomputes constants.
2. Then use `numpy` functions to construct matrix $A$.
3. Next calculate $\lambda_U$, $\lambda_L$.
4. In the `backward` function, we substitute the first and last element of payoff array with $\lambda_U$ and $\lambda_L$. Then use `np.solve` to solve the equation $Ax=b$.
5. Do `backward` for $N$ times.


```python
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

```

### (c)

The Crank-Nicolson finite different method averages the space derivatives at $i$ and $i+1$: 

$$
\begin{array}{l}
-\frac{V_{i+1, j}-V_{i, j}}{\Delta t} \\
=\frac{1}{2} \sigma^{2}\left(\frac{\left(V_{i+1, j+1}-2 V_{i+1, j}+V_{i+1, j-1}\right)+\left(V_{i, j+1}-2 V_{i, j}+V_{i, j-1}\right)}{2 \Delta x^{2}}\right) \\
\quad+\nu \left(\frac{\left(V_{i+1, j+1}-V_{i+1, j-1}\right)+\left(V_{l, j+1}-V_{i, j-1}\right)}{4 \Delta x}\right)-r\left(\frac{V_{i+1, j}+V_{i, j}}{2}\right)
\end{array}
$$

Which can be rewritten as:

$$
\begin{aligned}
p_{\mathrm{u}} V_{i, j+1}+p_{\mathrm{m}} V_{i, j}+p_{\mathrm{d}} V_{i, j-1} &=-p_{\mathrm{n}} V_{i+1, j+1}-\left(p_{\mathrm{m}}-2\right) V_{i+1, j}-p_{\mathrm{d}} V_{i+1, j-1} \\
p_{\mathrm{u}} &=-\frac{1}{4} \Delta t\left(\frac{\sigma^{2}}{\Delta x^{2}}+\frac{\nu}{\Delta x}\right) \\
p_{\mathrm{m}} &=1+\Delta t \frac{\sigma^{2}}{2 \Delta x^{2}}+\frac{r \Delta t}{2} \\
p_{\mathrm{d}} &=-\frac{1}{4} \Delta t\left(\frac{\sigma^{2}}{\Delta x^{2}}-\frac{\nu}{\Delta x}\right)
\end{aligned}
$$

So to implement it in python, the only difference is the $p_u$, $p_m$ $p_d$ and the matrix $A$:

```python
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

```

### (d)

