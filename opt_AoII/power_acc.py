import numpy as np
import sympy
import math
import matplotlib.pyplot as plt


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


#  ====================compare====================
def getmax(p, threshold):
    f1 = p / (d[0] ** alpha[0])
    f2 = p / (d[1] ** alpha[1])
    f3 = p / (d[2] ** alpha[2])
    f4 = p / (d[3] ** alpha[3])
    f5 = p / (d[4] ** alpha[4])
    f6 = p / (d[5] ** alpha[5])
    x1 = math.exp(-threshold * (
            sigma + h[1] * h[1] * f2 + h[2] * h[2] * f3 + h[3] * h[3] * f4 + h[4] * h[4] * f5 + h[5] * h[
        5] * f6) / f1)
    x2 = math.exp(-threshold * (
            sigma + h[2] * h[2] * f3 + h[3] * h[3] * f4 + h[4] * h[4] * f5 + h[5] * h[
        5] * f6) / f2)
    x3 = math.exp(-threshold * (
            sigma + h[3] * h[3] * f4 + h[4] * h[4] * f5 + h[5] * h[
        5] * f6) / f3)
    x4 = math.exp(-threshold * (
            sigma + h[4] * h[4] * f5 + h[5] * h[
        5] * f6) / f4)
    x5 = math.exp(-threshold * (
            sigma + h[5] * h[
        5] * f6) / f5)
    x6 = math.exp(-threshold * sigma / f6)
    y1 = accu(x1)
    y2 = accu(x2)
    y3 = accu(x3)
    y4 = accu(x4)
    y5 = accu(x5)
    y6 = accu(x6)
    return y1, y2, y3, y4, y5, y6


y1 = np.zeros(len(pmax))
y2 = np.zeros(len(pmax))
y3 = np.zeros(len(pmax))
y4 = np.zeros(len(pmax))
y5 = np.zeros(len(pmax))
y6 = np.zeros(len(pmax))

for i in range(len(pmax)):
    y1_value, y2_value, y3_value, y4_value, y5_value, y6_value = getmax(pmax[i], theta)
    y1[i] = y1_value
    y2[i] = y2_value
    y3[i] = y3_value
    y4[i] = y4_value
    y5[i] = y5_value
    y6[i] = y6_value

y_average = (y1 + y2 + y3 + y4 + y5 + y6) / 6

import matplotlib.pyplot as plt

plt.figure()
plt.plot(pmaxdB, y1, label='user1', linewidth=2)
plt.plot(pmaxdB, y2, label='user2', linewidth=2)
plt.plot(pmaxdB, y3, label='user3', linewidth=2)
plt.plot(pmaxdB, y4, label='user4', linewidth=2)
plt.plot(pmaxdB, y5, label='user5', linewidth=2)
plt.plot(pmaxdB, y6, label='user6', linewidth=2)
plt.plot(pmaxdB, y, label='our proposed', linestyle='--', linewidth=2)
plt.plot(pmaxdB, y_average, label='average', linestyle='--', linewidth=2)
font_xy = {'family': 'Times New Roman', 'size': '20'}
font_l = {'family': 'Times New Roman', 'size': '15'}
font1 = {'family': 'Times New Roman', 'weight': 'bold', 'size': '20'}
plt.xlabel('Transmit Power (dBm)', font1, labelpad=10)
plt.ylabel('Image Recognition Accuracy', font1, labelpad=10)
plt.ylim(0, 1)
plt.xlim(0, 40)
plt.xticks(family='Times New Roman', fontsize=20)
plt.yticks(family='Times New Roman', fontsize=20)
plt.legend(prop=font_l)
plt.tight_layout()
# plt.savefig('C:\\Users\\hp-pc\\Desktop\\ICC0927\\Fig7_compare.png')


plt.show()
