from new_algorithm import *
import math
import sympy
import matplotlib.pyplot as plt

AoI_1 = g[10][224]
AoI_2 = g[10][299]
AoI_3 = g[10][374]
AoI_4 = g[10][449]

g_value_miu = np.zeros(iter_miu + 1)
for count in range(iter_miu + 1):
    g_value_miu[count] = np.min(g[count])

AoI_m1 = g_value_miu[0]
AoI_m2 = g_value_miu[2]
AoI_m3 = g_value_miu[4]
AoI_m4 = g_value_miu[6]
AoI_m5 = g_value_miu[8]
AoI_m6 = g_value_miu[10]


def accu(sinr):
    a1 = 0.04592165
    a2 = 0.97662984
    c1 = 10.33466385
    c2 = -6.89161717
    y = a1 + (a2 - a1) / (1 + math.exp(-(c1 * sinr + c2)))
    return y


h = [0.9, 0.88, 0.86, 0.84, 0.82, 0.8]
d = [30, 28, 26, 24, 22, 20]
alpha = [3.7, 3.6, 3.5, 3.4, 3.3, 3.2]
sigma = 10 ** -6


def getx6(p, threshold):
    f1 = p / (d[0] ** alpha[0])
    x = sympy.Symbol('x')
    equation = (1 + h[5] * h[5] * x / sigma) * (1 + h[4] * h[4] * x / sigma) * (1 + h[3] * h[3] * x / sigma) * (
            1 + h[2] * h[2] * x / sigma) * (1 + h[1] * h[1] * x / sigma) * x - f1
    solution = sympy.solve(equation, x)
    f6 = solution[1]
    p6 = f6 * (d[5] ** alpha[5])
    f5 = (1 + h[5] * h[5] * f6 / sigma) * f6
    p5 = f5 * (d[4] ** alpha[4])
    f4 = (1 + h[5] * h[5] * f6 / sigma) * (1 + h[4] * h[4] * f6 / sigma) * f6
    p4 = f4 * (d[3] ** alpha[3])
    f3 = (1 + h[5] * h[5] * f6 / sigma) * (1 + h[4] * h[4] * f6 / sigma) * (1 + h[3] * h[3] * f6 / sigma) * f6
    p3 = f3 * (d[2] ** alpha[2])
    f2 = (1 + h[5] * h[5] * f6 / sigma) * (1 + h[4] * h[4] * f6 / sigma) * (1 + h[3] * h[3] * f6 / sigma) * (
            1 + h[2] * h[2] * f6 / sigma) * f6
    p2 = f2 * (d[1] ** alpha[1])
    f1 = (1 + h[5] * h[5] * f6 / sigma) * (1 + h[4] * h[4] * f6 / sigma) * (1 + h[3] * h[3] * f6 / sigma) * (
            1 + h[2] * h[2] * f6 / sigma) * (1 + h[1] * h[1] * f6 / sigma) * f6
    p1 = f1 * (d[0] ** alpha[0])
    x6 = math.exp(-threshold * sigma / f6)
    return x6


theta = 10 ** (-5 / 10)
pmaxdB = np.linspace(0, 40, 41)
pmax = [10 ** (item / 10) / 1000 for item in pmaxdB]

y = np.zeros(len(pmax))
for ite in range(len(pmax)):
    prob_x = getx6(pmax[ite], theta)
    acc = accu(prob_x)
    y[ite] = acc


# ====================图3,μ=4时不同的lamda值====================
# plt.figure()
# plt.plot(pmaxdB, (1-y)*AoI_1, label='$\lambda$ = 1.8')
# plt.plot(pmaxdB, (1-y)*AoI_2, label='$\lambda$ = 2.4')
# plt.plot(pmaxdB, (1-y)*AoI_3, label='$\lambda$ = 3.0')
# plt.plot(pmaxdB, (1-y)*AoI_4, label='$\lambda$ = 3.6')
# font_xy = {'family': 'Times New Roman', 'size': '20'}
# font1 = {'family': 'Times New Roman', 'weight': 'bold', 'size': '20'}
# plt.xlabel('Transmit Power (dBm)', font1, labelpad=10)
# plt.ylabel('AoII (s)', font1, labelpad=10)
# plt.xlim(0, 40)
# plt.ylim(0, 4.5)
# plt.xticks(family='Times New Roman', fontsize=20)
# plt.yticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5], family='Times New Roman', fontsize=20)
# plt.legend(prop=font_xy)
# plt.tight_layout()
# # plt.savefig('C:\\Users\\hp-pc\\Desktop\\ICC0927\\Fig8_lamda.png')
#
#
# plt.show()


# ====================图4,不同μ(取最优lamda)====================
plt.figure()
plt.plot(pmaxdB, (1-y)*AoI_m1, label='$\mu$ = 2', linewidth=2)
plt.plot(pmaxdB, (1-y)*AoI_m2, label='$\mu$ = 2.4', linewidth=2)
plt.plot(pmaxdB, (1-y)*AoI_m3, label='$\mu$ = 2.8', linewidth=2)
plt.plot(pmaxdB, (1-y)*AoI_m4, label='$\mu$ = 3.2', linewidth=2)
plt.plot(pmaxdB, (1-y)*AoI_m5, label='$\mu$ = 3.6', linewidth=2)
plt.plot(pmaxdB, (1-y)*AoI_m6, label='$\mu$ = 4', linewidth=2)
font_xy = {'family': 'Times New Roman', 'size': '20'}
font_l = {'family': 'Times New Roman', 'size': '15'}
font1 = {'family': 'Times New Roman', 'weight': 'bold', 'size': '20'}
plt.xlabel('Transmit Power (dBm)', font1, labelpad=10)
plt.ylabel('AoII (s)', font1, labelpad=10)
plt.xlim(0, 40)
plt.ylim(0, 6)
plt.xticks(family='Times New Roman', fontsize=20)
plt.yticks([0, 1, 2, 3, 4, 5, 6], family='Times New Roman', fontsize=20)
plt.legend(prop=font_l)
plt.tight_layout()
# plt.savefig('C:\\Users\\hp-pc\\Desktop\\ICC0927\\Fig8_miu.png')


plt.show()
