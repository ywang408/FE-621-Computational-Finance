---
title: finite diff method
author: You Wang
date: 03/27/2021
---

# PDE

For an option price under a certain stochastic model satisﬁes the following PDE:

$$
\frac{\partial V}{\partial t}+\alpha(S) \frac{\partial V}{\partial S}+\beta(S) \frac{\partial^{2} V}{\partial S^{2}}-r V=0
$$

Transform stock price process to return process, which is:

$$
x = \log S
$$

Using chain rule,

$$
\begin{aligned}
    \frac{\partial V}{\partial S} &= \frac{1}{S} \frac{\partial V}{\partial x}\\
    \frac{\partial^{2} V}{\partial S^{2}} &= \frac{1}{S^2} \left(\frac{\partial^{2} V}{\partial x^{2}} - \frac{\partial V}{\partial x}\right)
\end{aligned}
$$

Therefore the initial PDE becomes

$$
\frac{\partial V}{\partial t}+\left(\frac{\alpha(S)}{S} - \frac{\beta(S)}{S^2}\right) \frac{\partial V}{\partial S}+\frac{\beta(S)}{S^2} \frac{\partial^{2} V}{\partial S^{2}}-r V=0
$$

## Discretize the derivatives for explicit scheme

$$
\frac{V_{i+1, j}-V_{i, j}}{\Delta t}+a_{i+1, j} \frac{V_{i+1, j+1}-V_{i+1, j-1}}{2 \Delta x}+b_{i+1, j} \frac{V_{i+1, j+1}-2 V_{i, j}+V_{i+1, j-1}}{\Delta x^{2}}-r V_{i+1, j}=0
$$

where $a_{i,j} = \frac{\alpha(S_{i,j})}{S_{i,j}} - \frac{\beta(S_{i,j})}{S_{i,j}^2}$, $b_{i,j} = \frac{\beta(S_{i,j})}{S_{i,j}^2}$ at grid $(i,j)$.

Rearrange the equation:

$$\begin{aligned}
V_{i, j} &=p_{u} V_{i+1,j+1}+p_{m} V_{i+1, j}+p_{d} V_{i+1, j-1} \\
p_{u} &=\Delta t\left(\frac{b_{i+1, j}}{\Delta x^{2}}+\frac{a_{i+1, j}}{2 \Delta x}\right) \\
p_{m} &=1-\Delta t \frac{2 b_{i+1, j}}{\Delta x^{2}}-r \Delta t \\
p_{d} &=\Delta t\left(\frac{b_{i+1, j}}{\Delta x^{2}}-\frac{a_{i+1, j}}{2 \Delta x}\right)
\end{aligned}$$

## Discretize the derivatives for implicit scheme

$$
\frac{V_{i+1, j}-V_{i, j}}{\Delta t}+a_{i, j} \frac{V_{i, j+1}-V_{i, j-1}}{2 \Delta x}+b_{i, j} \frac{V_{i, j+1}-2 V_{i, j}+V_{i, j-1}}{\Delta x^{2}}-r V_{i, j}=0
$$

where $a_{i,j} = \frac{\alpha(S_{i,j})}{S_{i,j}} - \frac{\beta(S_{i,j})}{S_{i,j}^2}$, $b_{i,j} = \frac{\beta(S_{i,j})}{S_{i,j}^2}$ at grid $(i,j)$.

Rearrange the equation:

$$\begin{aligned}
V_{i+1, j} &=p_{u} V_{i,j+1}+p_{m} V_{i, j}+p_{d} V_{i, j-1} \\
p_{u} &=-\Delta t\left(\frac{b_{i, j}}{\Delta x^{2}}+\frac{a_{i, j}}{2 \Delta x}\right) \\
p_{m} &=1+\Delta t \frac{2 b_{i, j}}{\Delta x^{2}}+r \Delta t \\
p_{d} &=-\Delta t\left(\frac{b_{i, j}}{\Delta x^{2}}-\frac{a_{i, j}}{2 \Delta x}\right)
\end{aligned}$$
