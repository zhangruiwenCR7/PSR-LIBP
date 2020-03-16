import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

n = sys.argv[1]

train_loss, train_acc = [], []
val_loss, val_acc = [], []
test_loss, test_acc = [], []
with open('log/log'+n) as f:
    for line in f.readlines():
        line = line.strip().split()
        # print(line)
        if len(line)>1:
            if line[0] == '[Train]':
                train_loss.append(float(line[4]))
                train_acc.append(float(line[6]))
            elif line[0] == '[valid]':
                val_loss.append(float(line[4]))
                val_acc.append(float(line[6]))
            elif line[0] == '[test':
                test_loss.append(float(line[5]))
                test_acc.append(float(line[7]))
val_acc_sort = sorted(val_acc, reverse=True)
print(max(val_acc), val_acc.index(max(val_acc)))
plt.figure(figsize=(20,10))

plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)

plt.title('Phase Space Loss&Acc')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xlim([0,3000])
plt.ylim([0,1.2])
plt.grid()
plt.plot(train_loss, c='b', label='train_loss')
plt.plot(test_loss, c='m', label='test_loss')
plt.plot(val_loss, c='g', label='val_loss')
plt.legend(loc='upper left')
# plt.xlim([0,60])
ax = plt.twinx()
ax.set_ylim(0,1.2)
ax.set_ylabel('ACC')
ax.plot(train_acc, c='black', label='train_acc')
ax.plot(test_acc, c='y', label='test_acc')
ax.plot(val_acc, c='r',  label='val_acc')
plt.legend()
plt.savefig('loss'+n)