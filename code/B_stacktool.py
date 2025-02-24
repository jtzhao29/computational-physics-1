import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import expon
from scipy.stats import kstest
from scipy import stats

def random_block(L,gridll):
    """
    该函数随机挑选一个格子，放上去一个方块
    """
    x = np.random.randint(0,L)
    y = np.random.randint(0,L)
    gridll[x][y] += 1
    return gridll

def random_block_xy(L,gridll):
    """
    该函数随机挑选一个格子，放上去两个方块
    返回girdll和方块的xy坐标
    """
    x = np.random.randint(0,L)
    y = np.random.randint(0,L)
    gridll[x][y] += 1
    return gridll,x,y


def delete_block(L,gridll,x,y):
    """
    该函数执行“消消乐”的步骤，消去当前格子上的方块
    并记录⼀次消去
    这四个⽅块会移动累加到上下左右的4个格点（它们各⾃⽅块数+1）
    """
    gridll[x][y] -= 4
    if x >=1:
        gridll[x-1][y] += 1
    if x < L-1:
        gridll[x+1][y] += 1
    if y >=1:
        gridll[x][y-1] += 1
    if y < L-1:
        gridll[x][y+1] += 1
    return gridll



def exist_4_or_more(L,gridll):
    """
    这个函数判断当前场上是否存在大于4的方块
    """

    for i in range(L):
        for j in range(L):
            if gridll[i][j] >= 4:
                return [True,i,j]
    return [False,0,0]



def find_4_or_more(L,gridll,x,y,s):
    if gridll[x][y] < 4:    
        return 
    if gridll[x][y] >=4 :
        delete_block(L,gridll,x,y)
        s[0] += 1
        if x >=1:
            find_4_or_more(L,gridll,x-1,y,s)
        if x < L-1: 
            find_4_or_more(L,gridll,x+1,y,s)
        if y >=1:
            find_4_or_more(L,gridll,x,y-1,s)
        if y < L-1:
            find_4_or_more(L,gridll,x,y+1,s)
            





def main(total_time,L) :
    """
    这个函数模拟整个游戏

    """
    gridll = np.zeros((L,L))
    t = 0
    score_list = []
    average_density = []    
    while t < total_time:
        gridll,x,y = random_block_xy(L,gridll)
        s = [0]
        find_4_or_more(L,gridll,x,y,s)
        
        score_list.append(s[0])
        t += 1
        average_density.append(np.sum(gridll)/(L*L))
    return gridll,score_list,average_density




def plot_score_distribution(score_list, L):
    """
    绘制得分的频率分布直方图，并进行归一化
    """
    plt.figure(figsize=(10, 6))  
    n, bins, patches = plt.hist(score_list, bins=150, 
                                density=True, facecolor='g', alpha=0.75)  
   

    plt.xlabel('Score',fontsize=18)
    plt.ylabel('Frequency', fontsize=18) 
    plt.xlim(0,max(score_list)) 
    plt.title(f'Score Distribution (L={L})',fontsize=25) 
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)  

    plt.show()  

def plot_score_distribution_and_save(score_list, L):
    """
    绘制得分的频率分布直方图，并进行归一化
    """
    plt.figure(figsize=(10, 6))  
    wight_list = np.ones_like(score_list)/float(len(score_list))
    n, bins, patches = plt.hist(score_list, bins=12000, 
                                density=False, weights=wight_list, facecolor='g', alpha=0.75)  
   
    
    plt.xlabel('Score',fontsize=18)
    plt.ylabel('Frequency', fontsize=18) 
    # plt.xlim(0,40) 
    # plt.ylim(0,0.8)
    plt.title(f'Score Distribution (L={L})',fontsize=25) 
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)  
    save_path = f"./figure/B_2_{L}_new.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path) 
    plt.show() 

#     save_path = f"./figure/B_for_n_L.png"
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path)
#     plt.show()

   
# def plot_score_distribution_for_n_L(L_list, total_time):
#     score_list_for_L = {}
#     for L in L_list:
#         _, score_list, _ = main(total_time, L)  # 提取得分列表
#         score_list_for_L[f"score_{L}"] = score_list
#         plt.hist(score_list_for_L[f"score_{L}"], bins=3000,
#                  alpha=0.5, label=f"L = {L}", density=True)

#     plt.xlabel('Score')
#     plt.ylabel('Frequency')
#     plt.title('Score Distribution of multiple L')
#     plt.xlim(0, 50)
#     plt.legend()
#     plt.grid(True)

#     save_path = f"./figure/B_for_n_L.png"
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path)
#     plt.show()

def main_1(average_density,L):
    """
    这个函数画出平均密度随时间的变化图
    """
    plt.plot(average_density)
    plt.xlabel("Time",fontsize=18)
    plt.ylabel("Average Density",fontsize=18)
    plt.title(f"Average Density (L={L})",fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()

def average_density_fit(average_density):
    """
    这个函数拟合平均密度随时间的变化曲线
    """
    x = np.arange(2000)
    y = [average_density[i] for i in x]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    fit_line = slope*x + intercept
    print(f"Slope: {slope:.6f}, Intercept: {intercept:.2f}, R-squared: {r_value**2:.2f}")
    plt.plot(average_density, color='blue', label='Data Points')
    plt.plot(x, fit_line, color='red', label=f'Fit Line: y={slope:.6f}x+{intercept:.2f}')
    plt.xlabel("Time",fontsize=18)
    plt.ylabel("Average Density",fontsize=18)
    plt.title(f"Average Density (L={L})",fontsize=25)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(True, which="both", ls="--")


def plot_list_milvus_fit(score_list, L,len):
    """
    绘制得分的频率分布直方图，并进行归一化
    """
    scores = np.arange(1,len)
    frequencies = np.array([score_list.count(i) for i in range(1,len)])

    log_scores = np.log(scores+1)
    log_frequencies = np.log(frequencies)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_scores, log_frequencies)

    fit_line = slope*log_scores + intercept
     # 绘制log-log坐标下的直方图和拟合直线
    plt.figure(figsize=(10, 6))
    plt.scatter(log_scores, log_frequencies, label='Data Points')
    plt.plot(log_scores, fit_line, color='red', label=f'Fit Line: y={slope:.2f}x+{intercept:.2f}')
    
    plt.xlabel("Log(Score+1)", fontsize=18)
    plt.ylabel("Log(Frequency)", fontsize=18)
    plt.title(f"Log-Log Score Distribution (L={L})", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(True, which="both", ls="--")
    
    save_path = f"./figure/B_{L}_log_fit.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()

def plot_list_milvus_polyfit(score_list, L, len,degree=1):
    """
    绘制得分的频率分布直方图，并进行归一化，同时进行多项式拟合并绘制拟合曲线
    """
    scores = np.arange(len)
    frequencies = np.array([score_list.count(i) for i in range(len)])

    coefficients = np.polyfit(scores, frequencies)
    polynomial = np.poly1d(coefficients)

    fitted_values = polynomial(scores)

    fig, ax = plt.subplots()

    ax.scatter(scores, frequencies, color='blue', label='Original Data')

    ax.plot(scores, fitted_values, color='red', label=f'Fitted Polynomial ')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel("Score", fontsize=18)
    ax.set_ylabel("Frequency", fontsize=18)
    ax.set_title(f"Score Distribution and Fitting (L={L})", fontsize=18)

    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)

    ax.legend(fontsize=18)

    plt.show()


def plot_list_milvus(score_list, L,len):
    # 使用对数刻度绘制图形
    # fenbu = np.arange(len)
    fenbu = np.array([score_list.count(i) for i in range(len)])

    ax,fig = plt.subplots(figsize=(10, 6))
    ax.plot(fenbu)
    plt.xlabel("Score",fontsize=18)
    plt.ylabel("Frequency",fontsize=18)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.title(f"Score Distribution (L={L})",fontsize=25)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)
    save_path = f"./figure/B_{L}_log.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()


# 生成图像
# if __name__ == '__main__':
#     # L = 32
#     # total_time = 16000
#     # gridll,score_list,average_density = main(total_time,L)
#     # average_density_fit(average_density)
#     # # xmax = {5:20,10:35,15:70,20:100,25:200,30:200,32:200,35:250,40:250}
#     # plot_score_distribution(score_list, L)
#     L_list = [30,35,40,45,32]
#     L_list = [32]
#     # L_list = [32]
#     # print(f"L={L},the maximum score is:",max(score_list))
#     for L in L_list:
#         total_time = 16000
#         gridll,score_list,average_density = main(total_time,L)
#         print(f"L={L},the maximum score is:",max(score_list))
#         plot_score_distribution_and_save(score_list, L)


# if __name__ == '__main__':
#     L_list= [10,20,30,40,50]
#     total_time = 8000
#     plot_score_distribution_for_n_L(L_list,total_time)

# 验证指数分布
if __name__ == '__main__':
    L_list= [30,35,40,45,32]
    total_time = 16000
    for L in L_list:
        gridll,score_list,average_density = main(total_time,L)
        len =50
        plot_list_milvus_fit(score_list, L,len)
        # plot_list_milvus_polyfit(score_list, L,len,degree=3)


class Stack:
    def __init__(self):
        self.items = []
    
    def is_empty(self):
        return len(self.items) == 0
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("Stack is empty")
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("Stack is empty")
    
    def size(self):
        return len(self.items)

