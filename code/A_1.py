import numpy as np
import matplotlib.pyplot as plt

def f_1(mu,x):
    """
    定义第一问中的迭代方程
    """
    return 1-mu*x**2

def logistic_map(x0, mu, n_iter)->np.ndarray:  
    """
    表示映射的迭代函数。
    输入三个参数，输出各代的x值。

    参数设置:
        x0: 初值，(0, 1) 区间内的浮点数。
        mu: 参数，(0, 4) 区间内的浮点数。
        n_iter: 迭代次数。

    Returns:
        一个包含迭代结果的 NumPy 数组。
    """
    x = np.zeros(n_iter)
    x[0] = x0
    for i in range(n_iter - 1):
        x[i+1] = f_1(mu, x[i])
    return x

# def plot_logistic_map(mu_values, x0, n_iter):
#     """
#     绘制不同 mu 值下的 Logistic 映射结果。

#     Args:
#         mu_values: 一个包含多个 mu 值的列表或 NumPy 数组。
#         x0: 初值。
#         n_iter: 迭代次数。
#     """
#     plt.figure(figsize=(12, 6))
#     for mu in mu_values:
#         print
#         x = logistic_map(x0, mu, n_iter)
#         plt.plot(x, label=f"μ = {mu}")

#     plt.xlabel("Iteration")
#     plt.ylabel("x_n")
#     plt.title("Logistic Map")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

def plot_logistic_map(mu_values, x0, n_iter):
    """
    绘制不同 mu 值下的 Logistic 映射结果。
    将不同mu值的结果画到不同的子图中
    """
    num_plots = len(mu_values)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 6 * num_plots)) 

    for i, mu in enumerate(mu_values):
        x = logistic_map(x0, mu, n_iter)
        axes[i].plot(x, label=f"μ = {mu}")
        axes[i].xaxis.set_tick_params(labelsize=18)
        axes[i].yaxis.set_tick_params(labelsize=18)
        axes[i].set_xlabel("Iteration",fontsize=18)
        axes[i].set_ylabel("x_n",fontsize=18)
        axes[i].grid(True)
        axes[i].legend()
        
    plt.title(f"$x_n$ vs Iteration for func 1 for different $\mu$ values, $x_0$ = {x0}", 
              fontsize=19, y=10.5)
    plt.tight_layout()  # Adjust subplot parameters for a tight layout.
    plt.show()


# def plot_logistic_map_2(x_0value,mu, n_iter):
#     """
#     绘制不同初值下的 Logistic 映射结果。
#     """
#     num_plots = len(x_0value)
#     fig, axes = plt.subplots(num_plots, 1, figsize=(12, 6 * num_plots)) 

#     for i, x0 in enumerate(x_0value):
#         x = logistic_map(x0, mu, n_iter)
#         axes[i].plot(x, label=f"$x_0$ = {x0}")
#         axes[i].xaxis.set_tick_params(labelsize=18)
#         axes[i].yaxis.set_tick_params(labelsize=18)
#         axes[i].set_xlabel("Iteration",fontsize=18)
#         axes[i].set_ylabel("x_n",fontsize=18)
#         axes[i].grid(True)
#         axes[i].legend()

#     plt.title(f"$x_n$ vs Iteration for func 1 for different $\x_0$ values($\mu=${mu})",
#               fontsize=25, pad=20)
#     plt.tight_layout()  # Adjust subplot parameters for a tight layout.
#     plt.show()

def plot_logistic_map_2(x_0value, mu, n_iter):
    """
    绘制不同初值下的 Logistic 映射结果在一张图上。
    """
    fig, ax = plt.subplots(figsize=(12, 6))  # 创建一个图形和一个轴对象

    for x0 in x_0value:
        x = logistic_map(x0, mu, n_iter)
        ax.plot(x, label=f"$x_0$ = {x0}")  # 在同一个轴上绘制不同的初值结果

    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)
    ax.set_xlabel("Iteration", fontsize=18)
    ax.set_ylabel("x_n", fontsize=18)
    ax.grid(True)
    ax.legend()

    plt.title(f"$x_n$ vs Iteration for func 1 for different $x_0$ values($\mu={mu}$)", 
              fontsize=25, pad=20)
    plt.tight_layout()  # 调整子图参数以获得紧凑布局
    plt.show()



mu_values = [0.01,0.3,0.6,1,1.3,1.6,1.99] 
mu_values = [0.01,0.5,1,1.5,1.99]
x0 = 0.99
n_iter = 100

x_0value = [-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8]
x_0value = [-0.8,-0.6,-0.4,-0.2,0]
mu=0.5
plot_logistic_map_2(x_0value, mu, n_iter)


def plot_initial_condition_sensitivity(mu, x0_1, x0_2, n_iter):
    """
    绘制 Logistic 映射的初值敏感性。
    """
    x1 = logistic_map(x0_1, mu, n_iter)
    x2 = logistic_map(x0_2, mu, n_iter)

    plt.figure(figsize=(12, 6))
    plt.plot(x1, label=f"x0 = {x0_1}")
    plt.plot(x2, label=f"x0 = {x0_2}")

    plt.xlabel("Iteration",fontsize=18)
    plt.ylabel("x_n",fontsize=18)
    plt.title(f"Logistic Map - Initial Condition Sensitivity (μ = {mu})")
    plt.legend()
    plt.grid(True)
    plt.show()


mu = 3.9 
x0_1 = 0.2
x0_2 = 0.2001 
n_iter = 100

# plot_initial_condition_sensitivity(mu, x0_1, x0_2, n_iter)

