import matplotlib.pyplot as plt

# x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# y = [0.0483, 0.0520, 0.0698, 0.1049, 0.1797, 0.3547, 0.5979, 0.7908, 0.8801, 0.9619]
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.xlabel('Channel Condition')
# plt.ylabel('Task Accuracy')
#
# # 绘制描点画曲线
# plt.plot(x, y, marker='o', linestyle='-', color='blue')
#
# # 显示图形
# plt.show()

# ==================================================================================

import numpy as np
from scipy.optimize import curve_fit


def fit_function(x, Ak1, Ak2, Ck1, Ck2):
    return Ak1 + (Ak2 - Ak1)/(1 + np.exp(-(Ck1 * x + Ck2)))


x_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
y_data = np.array([0.0483, 0.0520, 0.0698, 0.1049, 0.1797, 0.3547, 0.5979, 0.7908, 0.8801, 0.9619])
popt, pcov = curve_fit(fit_function, x_data, y_data)
y_fit = fit_function(x_data, *popt)

plt.plot(x_data, y_fit, 'r-', label='Fitting Function', linewidth=2)
plt.scatter(x_data, y_data, edgecolor='black', facecolors='none', label='Original Data', linewidth=2)

font_xy = {'family': 'Times New Roman', 'size': '20'}
font1 = {'family': 'Times New Roman', 'weight': 'bold', 'size': '20'}
plt.xlabel('Successfully Decoding Accuracy at BS', font1, labelpad=10)
plt.ylabel('Image Recognition Accuracy', font1, labelpad=10)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xticks(family='Times New Roman', fontsize=20)
plt.yticks(family='Times New Roman', fontsize=20)
plt.legend(['Original Data', 'Fitting Function'], prop=font_xy)

plt.tight_layout()

# plt.savefig('C:\\Users\\hp-pc\\Desktop\\ICC0927\\Fig4_dot.png')
plt.show()
