sigma = 0.03
delta_x = 0.6
t = 4
r = 0
D = r - sigma ** 2 / 2
p_u = 0.5 * ((sigma ** 2 * t + D ** 2 * t ** 2) / delta_x ** 2 + D * t / delta_x)
p_m = 1 - (sigma ** 2 * t + D ** 2 * t ** 2) / delta_x ** 2
p_d = 0.5 * ((sigma ** 2 * t + D ** 2 * t ** 2) / delta_x ** 2 - D * t / delta_x)
print(p_u,p_m,p_d)