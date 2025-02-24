
import numpy as np
import matplotlib.pyplot as plt
from numpy import linspace

def f_2(mu,x):
    return np.cos(x) - mu*x**2

mu=0.5
x = linspace(-5, 5, 1000)
y = f_2(mu, x)
y2 = x

plt.plot(x, y)
plt.plot(x,y2)
plt.xlabel("x",fontsize=18)
plt.ylabel("f(x)",fontsize=18)
plt.title(f"f(x) for mu = {mu}",fontsize=25)
plt.grid()
plt.show()