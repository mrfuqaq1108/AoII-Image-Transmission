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


# pmaxdB = np.linspace(-20, 20, 41)
# pmax = [10 ** (item / 10) / 1000 for item in pmaxdB]


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


thetadB = np.linspace(-20, 20, 41)
theta = 10 ** (thetadB / 10)

pmaxdB = [0, 10, 20, 30, 40]
pmax = [10 ** (item / 10) / 1000 for item in pmaxdB]


def draw(pmax):
    y = np.zeros(len(thetadB))
    for ite in range(len(theta)):
        prob_x = getx6(pmax, theta[ite])
        acc = accu(prob_x)
        y[ite] = acc
    return y


y1 = draw(pmax[0])
y2 = draw(pmax[1])
y3 = draw(pmax[2])
y4 = draw(pmax[3])
y5 = draw(pmax[4])


plt.figure()
plt.plot(thetadB, y1, label='$p_{\max}$ = 0dB', linewidth=2)
plt.plot(thetadB, y2, label='$p_{\max}$ = 10dB', linewidth=2)
plt.plot(thetadB, y3, label='$p_{\max}$ = 20dB', linewidth=2)
plt.plot(thetadB, y4, label='$p_{\max}$ = 30dB', linewidth=2)
plt.plot(thetadB, y5, label='$p_{\max}$ = 40dB', linewidth=2)
font_xy = {'family': 'Times New Roman', 'size': '20'}
font1 = {'family': 'Times New Roman', 'weight': 'bold', 'size': '20'}
plt.xlabel('SINR Threshold (dB)', font1, labelpad=10)
plt.ylabel('Image Recognition Accuracy', font1, labelpad=10)
plt.ylim(0, 1)
plt.xlim(-20, 20)
plt.xticks(family='Times New Roman', fontsize=20)
plt.yticks(family='Times New Roman', fontsize=20)
plt.legend(prop=font_xy)
plt.tight_layout()
# plt.savefig('C:\\Users\\hp-pc\\Desktop\\ICC0927\\Fig6_SINR.png')

plt.show()
