from B_tool import *

def plot_score_distribution(score_list, L):
    """
    绘制得分的频率分布直方图，并进行归一化
    """
    plt.figure(figsize=(10, 6))  
    n, bins, patches = plt.hist(score_list, bins=30, 
                                density=True, facecolor='g', alpha=0.75)  
   

    plt.xlabel('Score')
    plt.ylabel('Frequency') 
    plt.title(f'Score Distribution (L={L})') 
    plt.grid(True)  

    plt.show()  


L = 32
total_time = 20000
gridll, score_list, average_density = main(total_time, L)
plot_score_distribution(score_list, L)
