import numpy as np
import matplotlib.pyplot as plt


def function(lamda0, miu, M):
    g = 1 / lamda0 + (1 / (miu - lamda0 * M) + (M - 1) * lamda0 / miu) * (lamda0 / (miu - (M - 1) * lamda0)) + (
            (M - 1) * lamda0 / miu / (miu - (M - 1) * lamda0)) * (
                (miu - M * lamda0) / (miu - (M - 1) * lamda0)) + 1 / miu
    return g


# ==========Initial==========
M = 6

miu_max = 4
miu_min = 2

iter_miu = 10
iter_lamda = 500

step_miu = (miu_max - miu_min) / iter_miu

# miu共有iter_miu+1个点, lamda0共有iter_lamda-1个点
# 矩阵形状为(iter_miu + 1, iter_lamda -1)
g = np.zeros((iter_miu + 1, iter_lamda - 1))

for iter_u in range(iter_miu + 1):
    miu_temp = miu_min + iter_u * step_miu
    lamda0_max = miu_temp / M
    step_lamda = lamda0_max / iter_lamda
    lamda0_min = step_lamda
    for iter_l in range(iter_lamda - 1):
        lamda0_temp = lamda0_min + iter_l * step_lamda
        g_value = function(lamda0_temp, miu_temp, M)
        g[iter_u][iter_l] = g_value

#   求解最小值
# print(g)
opt_g_value = np.min(g)  # 输出最小值
# print(opt_g_value)
miu_index, lamda0_index = np.where(g == opt_g_value)
miu_value = miu_min + step_miu * miu_index.item()  # 最小值时miu的值
lamda_value = miu_value / M / iter_lamda * (1 + lamda0_index.item())  # 最小值时lamda0的值
# print('miu = ', miu_value)
# print('lamda = ', lamda_value)
# print('value = ', function(lamda_value, miu_value, M))

# ====================画图1,g~lamda,不同miu的情况下====================

miu_index_2 = miu_min + step_miu * 2
miu_index_4 = miu_min + step_miu * 4
miu_index_6 = miu_min + step_miu * 6
miu_index_8 = miu_min + step_miu * 8

lamda_vector_2 = np.linspace(miu_index_2 / M / iter_lamda, miu_index_2 / M * (1 - 1 / iter_lamda), iter_lamda - 1)
lamda_vector_4 = np.linspace(miu_index_4 / M / iter_lamda, miu_index_4 / M * (1 - 1 / iter_lamda), iter_lamda - 1)
lamda_vector_6 = np.linspace(miu_index_6 / M / iter_lamda, miu_index_6 / M * (1 - 1 / iter_lamda), iter_lamda - 1)
lamda_vector_8 = np.linspace(miu_index_8 / M / iter_lamda, miu_index_8 / M * (1 - 1 / iter_lamda), iter_lamda - 1)

# plt.figure()
# plt.plot(lamda_vector_2*6, g[2], label='$\mu$=2.4', linewidth=2)
# plt.plot(lamda_vector_4*6, g[4], label='$\mu$=2.8', linewidth=2)
# plt.plot(lamda_vector_6*6, g[6], label='$\mu$=3.2', linewidth=2)
# plt.plot(lamda_vector_8*6, g[8], label='$\mu$=3.6', linewidth=2)
# font_xy = {'family': 'Times New Roman', 'size': '20'}
# font1 = {'family': 'Times New Roman', 'weight': 'bold', 'size': '20'}
# plt.xlabel('$\lambda$', font1, labelpad=10)
# plt.ylabel('AoI (s)', font1, labelpad=10)
# plt.xlim(0, 3.6)
# plt.ylim(3, 30)
# plt.xticks([0, 0.6, 1.2, 1.8, 2.4, 3, 3.6], family='Times New Roman', fontsize=20)
# plt.yticks(family='Times New Roman', fontsize=20)
# plt.legend(prop=font_xy)
#
# plt.tight_layout()
# # plt.savefig('C:\\Users\\hp-pc\\Desktop\\ICC1013\\Fig5_new.png')
#
# plt.show()

# ====================画图2,g~miu====================

miu_vector = np.linspace(miu_min, miu_max, iter_miu + 1)
g_value_miu = np.zeros(iter_miu + 1)
for count in range(iter_miu + 1):
    g_value_miu[count] = np.min(g[count])

# plt.figure()
# plt.plot(miu_vector, g_value_miu, linewidth=2)
# font_xy = {'family': 'Times New Roman', 'size': '20'}
# font1 = {'family': 'Times New Roman', 'weight': 'bold', 'size': '20'}
# plt.xlabel('$\mu$', font1, labelpad=10)
# plt.ylabel('AoI (s)', font1, labelpad=10)
# plt.xlim(2, 4)
# plt.ylim(3, 6)
# plt.xticks([2, 2.5, 3, 3.5, 4], family='Times New Roman', fontsize=20)
# plt.yticks(family='Times New Roman', fontsize=20)
# plt.tight_layout()
# # plt.savefig('C:\\Users\\hp-pc\\Desktop\\ICC0927\\Fig5_miu.png')
#
#
# plt.show()
