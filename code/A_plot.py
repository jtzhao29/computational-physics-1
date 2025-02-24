
# import numpy as np
# import matplotlib.pyplot as plt
# from numpy import linspace

# def f_2(mu,x):
#     return np.cos(x) - mu*x**2

# mu=0.5
# x = linspace(-5, 5, 1000)
# y = f_2(mu, x)
# y2 = x

# plt.plot(x, y)
# plt.plot(x,y2)
# plt.xlabel("x",fontsize=18)
# plt.ylabel("f(x)",fontsize=18)
# plt.title(f"f(x) for mu = {mu}",fontsize=25)
# plt.grid()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# 定义函数 f(x) 和其导数 f'(x)
def f(x, mu):
    return x - np.cos(x) - mu * x**2

def df(x, mu):
    return 1 + np.sin(x) - 2 * mu * x

# 定义方程组：f(x) = 0 且 f'(x) = 0
def equations(vars):
    x, mu = vars
    return [x - np.cos(x) - mu * x**2,  # f(x) = 0
            1 + np.sin(x) - 2 * mu * x]  # f'(x) = 0

# 使用 fsolve 求解方程组
x_tangent, mu = fsolve(equations, [1.0, 0.5])  # 初始猜测值 [x, mu]

print(f"mu = {mu:.6f}")
print(f"切点坐标: ({x_tangent:.6f}, 0)")

# 绘制函数图像
x = np.linspace(-1, 3, 1000)
y = f(x, mu)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label=f'$x - \cos(x) - {mu:.6f}x^2$')
plt.plot(x_tangent, 0, 'ro', label='tangent point')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)
plt.legend()
plt.title(f'plot of $g(x)$ ($\mu = {mu:.6f}$)',
           fontsize=25)
plt.xlabel('$x$', fontsize=18)
plt.ylabel('$g(x)$', fontsize=18)

plt.show()