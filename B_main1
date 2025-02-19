from B_tool import *

def main_1(average_densityL):
    """
    这个函数画出平均密度随时间的变化图
    """
    plt.plot(average_density)
    plt.xlabel("Time")
    plt.ylabel("Average Density")
    plt.show()

if __name__ == '__main__':
    L = 32
    total_time = 10000
    gridll,score_list,average_density = main(total_time,L)
    main_1(average_density)
    