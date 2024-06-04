import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_lists(list1, list2, list3, list4, list5, list6, list7, list8, list9):
    sns.set(style="whitegrid", context="talk")
    plt.figure(figsize=(16, 10), dpi=300)
    colors = sns.color_palette("husl", 9)
    lines = ["-", "-", "-", "-", "-", "-", "-", "-", "-"]
    markers = ["o", "s", "p", "*", "h", "H", "X", "D", "<"]
    sns.lineplot(data=np.array(list1), marker=markers[0], linestyle=lines[0], color=colors[0], label='0.1,0.9',
                 antialiased=True)
    sns.lineplot(data=np.array(list2), marker=markers[1], linestyle=lines[1], color=colors[1], label='0.2,0.8',
                 antialiased=True)
    sns.lineplot(data=np.array(list3), marker=markers[2], linestyle=lines[2], color=colors[2], label='0.3,0.7',
                 antialiased=True)
    sns.lineplot(data=np.array(list4), marker=markers[3], linestyle=lines[3], color=colors[3], label='0.4,0.6',
                 antialiased=True)
    sns.lineplot(data=np.array(list5), marker=markers[4], linestyle=lines[4], color=colors[4], label='0.5,0.5',
                 antialiased=True)
    sns.lineplot(data=np.array(list6), marker=markers[5], linestyle=lines[5], color=colors[5], label='0.6,0.4',
                 antialiased=True)
    sns.lineplot(data=np.array(list7), marker=markers[6], linestyle=lines[6], color=colors[6], label='0.7,0.3',
                 antialiased=True)
    sns.lineplot(data=np.array(list8), marker=markers[7], linestyle=lines[7], color=colors[7], label='0.8,0.2',
                 antialiased=True)
    sns.lineplot(data=np.array(list9), marker=markers[8], linestyle=lines[8], color=colors[8], label='0.9,0.1',
                 antialiased=True)
    plt.legend(loc='upper left', fancybox=True, shadow=True)
    plt.title('Predictive performance under different parameters', fontsize=30)
    plt.xlabel('Sample pairs', fontsize=30)
    plt.ylabel('Prediction error count', fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    plt.show()


list_results = [
    [0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8,
     8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 13, 13, 13, 13, 13, 14, 14, 14,
     14, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 18, 18, 18, 18, 18, 18, 19, 19, 19, 20, 20, 21, 21, 21,
     21, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27, 28, 28, 29, 29, 29, 30, 31,
     31, 32, 32, 32, 32, 33, 33, 33, 33, 34, 34, 35, 35, 36, 37, 37, 37, 38, 38, 39, 39, 39, 39, 39, 40, 40, 40, 41, 41,
     41, 42, 42, 42, 42, 42, 43, 43, 43, 43, 44, 44, 44, 44, 45, 45, 45, 45, 46, 46, 46, 47, 47, 48, 48, 48, 48, 48, 48,
     49, 49, 49, 50, 50],
    [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9, 9, 9, 9,
     10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14,
     14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16,
     17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20,
     20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23,
     24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 26, 26, 27, 28, 28, 28, 28, 28, 29, 29, 29, 30, 30, 30, 30, 30, 30, 30, 31,
     31, 31, 31, 31, 31, 32, 32],
    [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7,
     8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15,
     15, 16, 16, 16, 16, 16, 17, 17, 17, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 21,
     21, 22, 23, 23, 23, 24, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28,
     28, 29, 29, 29, 30, 30, 30, 30, 31, 31, 31, 31, 31, 32, 32, 32, 32, 33, 34, 34, 34, 35, 36, 36, 37, 37, 38, 38, 38,
     38, 38, 39, 39, 39, 39, 39, 40, 40, 40, 40, 41, 41, 41, 41, 42, 42, 42, 42, 42, 43, 43, 44, 44, 44, 44, 44, 44, 45,
     45, 45, 45, 45, 45],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6,
     6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12,
     12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16,
     16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19,
     19, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23,
     23, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 28, 28, 28, 28,
     28],
    [0, 0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 13, 14, 14,
     14, 14, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 18, 18, 19, 19, 20, 21, 21, 21, 21, 22, 23, 23, 23, 24, 25, 25, 25,
     25, 25, 26, 26, 26, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 31, 31, 31, 31, 31, 32, 32, 32, 33, 33, 33,
     34, 35, 35, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 38, 38, 39, 39, 40, 40, 40, 40, 40, 41, 41, 42, 43, 43, 43, 43,
     43, 44, 44, 44, 44, 44, 45, 45, 45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 49, 50, 50, 50, 50, 51, 51,
     51, 51, 52, 52, 53, 53, 53, 54, 54, 54, 55, 55, 56, 56, 56, 57, 57, 57, 58, 58, 58, 59, 59, 59, 60, 60, 61, 61, 61,
     62, 63, 64, 64, 64, 65, 65, 66, 67, 68],
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7, 8, 9, 9, 10, 10, 11, 11, 11, 11, 12, 12, 13, 13,
     13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 19, 19,
     20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 25, 25, 25, 25,
     25, 25, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 30,
     30, 30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 33, 33, 33, 33, 34, 34, 34, 35, 35, 35, 35, 35,
     36, 36, 36, 36, 36, 37, 37, 37, 38, 38, 38, 38, 39, 39, 39, 40, 41, 41, 41, 41, 42, 42, 42, 43, 43, 44, 45, 45, 46,
     46, 46, 47, 47, 47, 47, 48, 49, 49, 49],
    [0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8, 8,
     8, 9, 9, 10, 10, 10, 10, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15,
     16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 21, 21, 21,
     21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 25, 25, 25, 25, 26, 26, 26, 27, 27, 27, 27, 27, 28,
     28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 31, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 34,
     34, 34, 34, 35, 35, 35, 36, 36, 36, 36, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42,
     42, 42, 42, 42, 42, 42, 42],
    [0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6, 7, 7, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12,
     12, 12, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 17, 17, 17,
     18, 18, 19, 19, 19, 19, 19, 19, 20, 20, 20, 21, 21, 22, 22, 22, 23, 23, 24, 24, 25, 25, 25, 25, 26, 27, 27, 27, 27,
     28, 28, 28, 28, 29, 29, 30, 30, 30, 30, 30, 30, 31, 31, 32, 32, 32, 32, 32, 33, 33, 34, 34, 35, 35, 35, 36, 36, 36,
     36, 37, 37, 37, 37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 41, 41, 41, 41, 42, 42, 42, 43, 43, 44, 44,
     44, 44, 44, 44, 45, 45, 45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 48, 48, 49, 49, 49, 49, 50, 51, 51, 51, 51, 51,
     51, 51, 52, 52, 52, 53, 54, 54, 54],
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 10, 11, 11,
     12, 12, 12, 13, 13, 14, 15, 16, 16, 16, 16, 16, 17, 17, 17, 18, 18, 19, 19, 19, 20, 21, 21, 21, 22, 22, 22, 22, 22,
     23, 23, 23, 24, 24, 24, 25, 25, 25, 25, 26, 27, 27, 27, 27, 27, 28, 28, 28, 29, 29, 30, 30, 30, 30, 31, 31, 32, 32,
     32, 32, 32, 33, 33, 34, 34, 34, 34, 34, 35, 35, 36, 37, 37, 38, 38, 38, 39, 39, 39, 39, 40, 40, 41, 41, 41, 42, 43,
     44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 48, 48, 48, 48, 49, 49, 49, 50, 50, 51, 51, 51, 51, 51, 52, 52, 53, 53, 53,
     54, 54, 54, 54, 55, 55, 55, 56, 56, 57, 57, 57, 58, 58, 59, 59, 60, 61, 61, 61, 62, 63, 64, 65, 66, 66, 67, 67, 68,
     69, 70, 70, 70, 71, 71, 72, 72]
]

plot_lists(list_results[0], list_results[1], list_results[2], list_results[3], list_results[4], list_results[5],
           list_results[6], list_results[7], list_results[8])