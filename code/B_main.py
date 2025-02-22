import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from B_tree import main

def plot_log_log_pdf(score_lists, labels):
    """
    绘制多个得分列表的概率密度函数的对数-对数图。

    参数:
    score_lists (list of lists): 包含多个得分列表的列表。
    labels (list of str): 每个得分列表对应的标签。
    """
    fig, ax = plt.subplots()

    for score_list, label in zip(score_lists, labels):
        # 计算概率密度函数
        kde = gaussian_kde(score_list)
        scores = np.linspace(min(score_list), max(score_list), 1000)
        density = kde(scores)

        # 对得分和概率取对数
        log_scores = np.log(scores+1)
        log_density = np.log(density)

        # 绘制对数-对数图
        ax.plot(log_scores, log_density, label=label)

    ax.set_xlabel("Log(Score+1)", fontsize=18)
    ax.set_ylabel("Log(Density)", fontsize=18)
    ax.set_title("Log-Log Plot of Score Density", fontsize=25)

    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)

    ax.legend(fontsize=18)

    plt.show()


if __name__ == '__main__':
    total_time = 100000
    L_values = [30, 35, 40, 45]
    L_values = [32]
    score_lists = []
    labels = []

    for L in L_values:
        gridll, score_list, average_density = main(total_time, L)
        score_lists.append(score_list)
        labels.append(f'L={L}')

    plot_log_log_pdf(score_lists, labels)