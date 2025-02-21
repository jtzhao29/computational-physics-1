import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import expon
from scipy.stats import kstest

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
    n, bins, patches = plt.hist(score_list, bins=1200, 
                                density=False, weights=wight_list, facecolor='g', alpha=0.75)  
   
    
    plt.xlabel('Score',fontsize=18)
    plt.ylabel('Frequency', fontsize=18) 
    plt.xlim(0,40) 
    plt.ylim(0,0.8)
    plt.title(f'Score Distribution (L={L})',fontsize=25) 
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)  
    save_path = f"./figure/B_2_{L}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path) 
    plt.show() 
   
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


def main_1(average_density):
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


# # 生成图像
# if __name__ == '__main__':
#     # L = 30
#     # total_time = 16000
#     # gridll,score_list,average_density = main(total_time,L)
#     # main_1(average_density)
#     # # xmax = {5:20,10:35,15:70,20:100,25:200,30:200,32:200,35:250,40:250}
#     # plot_score_distribution(score_list, L)
#     L_list = [30,35,40,45,32]
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
    L = 32
    total_time = 16000
    gridll,score_list,average_density = main(total_time,L)
    # 验证指数分布
    data = np.array(score_list)
    df = 4
    result = kstest(data, 'chi2',args = (df,))
    print(result)