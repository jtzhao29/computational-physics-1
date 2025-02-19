import numpy as np
import matplotlib.pyplot as plt

def logistic_map(x0, mu, n_iter):
    """
    Logistic 映射的迭代函数。

    Args:
        x0: 初值，(0, 1) 区间内的浮点数。
        mu: 参数，(0, 4) 区间内的浮点数。
        n_iter: 迭代次数。

    Returns:
        一个包含迭代结果的 NumPy 数组。
    """
    x = np.zeros(n_iter)
    x[0] = x0
    for i in range(n_iter - 1):
        x[i+1] = mu * x[i] * (1 - x[i])
    return x

def plot_logistic_map(mu_values, x0, n_iter):
    """
    绘制不同 mu 值下的 Logistic 映射结果。

    Args:
        mu_values: 一个包含多个 mu 值的列表或 NumPy 数组。
        x0: 初值。
        n_iter: 迭代次数。
    """
    plt.figure(figsize=(12, 6))
    for mu in mu_values:
        x = logistic_map(x0, mu, n_iter)
        plt.plot(x, label=f"μ = {mu}")

    plt.xlabel("Iteration")
    plt.ylabel("x_n")
    plt.title("Logistic Map")
    plt.legend()
    plt.grid(True)
    plt.show()

# 示例
mu_values = [0.5, 2.0, 3.2, 3.9]  # 选择几个有代表性的 mu 值
x0 = 0.2  # 初始值
n_iter = 100

plot_logistic_map(mu_values, x0, n_iter)

# 初值敏感性分析
def plot_initial_condition_sensitivity(mu, x0_1, x0_2, n_iter):
    """
    绘制 Logistic 映射的初值敏感性。
    """
    x1 = logistic_map(x0_1, mu, n_iter)
    x2 = logistic_map(x0_2, mu, n_iter)

    plt.figure(figsize=(12, 6))
    plt.plot(x1, label=f"x0 = {x0_1}")
    plt.plot(x2, label=f"x0 = {x0_2}")

    plt.xlabel("Iteration")
    plt.ylabel("x_n")
    plt.title(f"Logistic Map - Initial Condition Sensitivity (μ = {mu})")
    plt.legend()
    plt.grid(True)
    plt.show()

# 示例
mu = 3.9  # 在混沌区域选择一个 mu 值
x0_1 = 0.2
x0_2 = 0.2001  # 稍微改变一下初始值
n_iter = 100

plot_initial_condition_sensitivity(mu, x0_1, x0_2, n_iter)