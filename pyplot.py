# 1_4_3_gradient_method_quadratic
# ! python3

from numpy import *
import numpy as np
import scipy.io as scio
from matplotlib import pyplot as plt
import matplotlib.pyplot as pl
from matplotlib.ticker import MultipleLocator, FuncFormatter

alarmNum = []
alarmRatio = []
for i in range(1, 30):
    dataResult = [100 + i, i]
    alarmNum.append(dataResult)

plt.figure(figsize=(30, 10))
plt.plot(alarmNum, label="psa_oc_result_analysis")
ax = plt.gca()
# 主刻度为1的倍数
ax.xaxis.set_major_locator(MultipleLocator(1))
plt.xlabel('April')
plt.ylabel('Camera1 human pic num')
plt.grid()
plt.legend()
plt.savefig("alarmNum.pdf")
plt.show()

for i in range(1, 30):
    dataRatio = i / (100 + i)
    alarmRatio.append(dataRatio)

plt.figure(figsize=(30, 10))
plt.plot(alarmRatio, label="psa_oc_result_analysis")
ax = plt.gca()
# 主刻度为1的倍数
ax.xaxis.set_major_locator(MultipleLocator(1))
plt.xlabel('April')
plt.ylabel('Camera1 human recognise ration')
plt.grid()
plt.legend()
plt.savefig("alarmRatio.pdf")
plt.show()

data = np.array([alarmNum, alarmRatio])
dataFile = './psa_oc_result_analysis.mat'
scio.savemat(dataFile, {'data': data})
