from B_tree import main,main_1

L = 256
total_time = 1000000

gridll,score_list,average_density = main(total_time,L)

main_1(average_density,L)

