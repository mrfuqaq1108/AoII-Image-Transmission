import numpy as np
import torch
import matplotlib.pyplot as plt

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

dataset = 'raw-img'

history = torch.load('raw-img_history.pt')

history = np.array(history)
print(history)

# plt.plot(history[:, 0:2], linewidth=2)
# font_xy = {'family': 'Times New Roman', 'size': '20'}
# font1 = {'family': 'Times New Roman', 'weight': 'bold', 'size': '20'}
# plt.legend(['Training Loss', 'Validation Loss'], prop=font_xy)
# plt.xlabel('Epoch Number', font1, labelpad=10)
# plt.ylabel('Loss', font1, labelpad=10)
# plt.xlim(0, 30)
# plt.ylim(0, 0.5)
# plt.xticks(family='Times New Roman', fontsize=20)
# plt.yticks(family='Times New Roman', fontsize=20)
#
# plt.tight_layout()
#
# # plt.savefig('C:\\Users\\hp-pc\\Desktop\\ICC0927\\Fig3_loss.png')
# plt.show()

# plt.plot(history[:, 2:4], linewidth=2)
# font_xy = {'family': 'Times New Roman', 'size': '20'}
# font1 = {'family': 'Times New Roman', 'weight': 'bold', 'size': '20'}
# plt.legend(['Training Accuracy', 'Validation Accuracy'], prop=font_xy)
# plt.xlabel('Epoch Number', font1, labelpad=10)
# plt.ylabel('Accuracy', font1, labelpad=10)
# plt.xlim(0, 30)
# plt.ylim(0.8, 1)
# plt.xticks(family='Times New Roman', fontsize=20)
# plt.yticks([0.8, 0.85, 0.9, 0.95, 1], family='Times New Roman', fontsize=20)
# plt.tight_layout()
#
# # plt.savefig('C:\\Users\\hp-pc\\Desktop\\ICC0927\\Fig3_acc.png')
# plt.show()
