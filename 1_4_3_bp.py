# 1_4_3_gradient_method_quadratic
# ! python3

from numpy import *
import numpy as np
import scipy.io as scio
from matplotlib import pyplot as plt

x1 = [29, 43, 50, 36, 2, 29, 94, 45, 83, 93, 41, 76, 39, 91, 45, 28, 28, 39, 57,
      72, 12, 38, 52, 83, 32, 15, 100, 49, 83, 93, 50, 45, 72, 56, 36, 43, 11,
      23, 67, 67, 2, 86, 14, 92, 51, 52, 94, 55, 87, 40, 14, 30, 24, 22, 17, 22,
      7, 43, 71, 23, 32, 33, 53, 32, 27, 94, 19, 54, 76, 50, 67, 53, 2, 45, 73, 98,
      19, 15, 9, 24, 64, 51, 62, 86, 58, 32, 53, 45, 52, 85, 84, 29, 83, 67, 91, 45, 71, 8, 76, 99]
x2 = [54, 47, 65, 79, 56, 80, 55, 65, 85, 67, 67, 42, 45, 56, 21, 67, 72, 50, 33, 2,
      81, 55, 66, 13, 30, 63, 55, 7, 39, 19, 61, 43, 35, 69, 45, 69, 39, 83, 50, 60,
      44, 10, 56, 99, 59, 17, 66, 68, 41, 83, 26, 42, 79, 0, 14, 48, 44, 96, 65, 37,
      61, 85, 72, 40, 24, 82, 0, 44, 58, 35, 43, 28, 67, 58, 37, 28, 34, 40, 61, 45,
      22, 28, 60, 91, 81, 51, 83, 75, 97, 91, 58, 83, 68, 57, 55, 38, 87, 59, 100, 80]
x3 = [98, 56, 31, 78, 30, 90, 73, 52, 37, 21, 93, 97, 25, 60, 90, 66, 28, 69, 46, 67,
      32, 56, 95, 6, 1, 86, 52, 89, 61, 26, 82, 97, 52, 43, 39, 95, 59, 2, 22, 6, 83,
      91, 0, 51, 76, 94, 45, 37, 11, 40, 9, 12, 7, 19, 60, 38, 17, 76, 55, 89, 91, 44,
      18, 55, 24, 73, 32, 29, 75, 9, 69, 95, 84, 69, 58, 59, 93, 37, 1, 57, 84, 75, 17,
      70, 40, 6, 86, 11, 71, 64, 95, 19, 21, 17, 3, 79, 33, 41, 19, 42]
y1 = [973, 202, 77, 541, 59, 796, 429, 187, 131, 63, 853, 938, 40, 256, 738, 335, 77, 357,
      114, 308, 100, 210, 906, 10, 12, 677, 181, 710, 250, 30, 594, 936, 160, 133, 83, 909,
      222, 71, 42, 43, 591, 763, 33, 240, 479, 839, 144, 102, 27, 137, 9, 22, 65, 9, 220, 80,
      25, 535, 216, 721, 794, 161, 63, 186, 22, 466, 35, 49, 463, 18, 354, 871, 638, 367, 216,
      223, 818, 68, 38, 208, 604, 435, 47, 434, 135, 29, 710, 62, 457, 353, 899, 79, 64, 44,
      39, 512, 119, 105, 114, 148]
y2 = [63, 107, 170, 117, 34, 97, 868, 139, 648, 851, 123, 466, 82, 791, 105, 73, 77, 91, 201,
      380, 71, 91, 194, 574, 42, 52, 1035, 127, 593, 811, 170, 119, 391, 228, 71, 137, 22, 81,
      328, 337, 28, 646, 34, 882, 175, 153, 879, 216, 676, 137, 10, 46, 77, 13, 13, 37, 21, 179,
      406, 35, 79, 113, 203, 54, 28, 905, 10, 180, 480, 138, 326, 166, 53, 132, 409, 955, 28, 23,
      38, 40, 275, 148, 276, 726, 265, 59, 226, 148, 242, 703, 636, 95, 620, 335, 784, 113,
      437, 39, 541, 1038]
s1 = []
s2 = []


def datafile():
    # 使用sigmoid函数，对输出y1, y2进行尺度高速至(0,1)范围
    for i in range(len(y1)):
        s1.append(1 / (1 + exp(-y1[i])))
        s2.append(1 / (1 + exp(-y2[i])))


datafile()

data = np.array([x1, x2, x3, s1, s2])
# dataFile = './bp_data.mat'
# scio.savemat(dataFile, {'data': data})

# -------------函数说明----------------
#     交换地点顺序
#        输入变量：
#                training_example:待训练的数据
#                     eta： 学习速率
# ---------------------------------------
m, n = shape(data)

# 初始化权值矩阵-0.5~0.5之间
w = np.random.rand(2, 3) - 0.5
v = np.random.rand(3, 2) - 0.5
u = np.random.rand(2, 3) - 0.5

hidden1 = [0, 0]
hidden2 = [0, 0, 0]
o = [0, 0]

delta3 = [0, 0]
delta2 = [0, 0, 0]
delta1 = [0, 0]

eta = 0.9
sigma = []


def BP_BACK(training_example, eta):
    # 对每列求输入与输出值
    # for num in range(n):
    for num in range(n):
        one_sample = data[:, num]
        # get input: x1, x2, x3
        x = one_sample[0:3]
        # get output: s1, s2
        y = one_sample[3:5]

        # get sum of layer 1
        net2 = np.dot(w, x)
        # process output hidden with sigmoid function
        for i in range(2):
            hidden1[i] = 1 / (1 + exp(-net2[i]))

        # get sum of layer 2
        net3 = np.dot(v, np.array(hidden1).transpose())
        # process output hidden with sigmoid function
        for i in range(3):
            hidden2[i] = 1 / (1 + exp(-net3[i]))

        # get sum of layer 3
        net4 = np.dot(u, np.array(hidden2).transpose())
        # process output hidden with sigmoid function, then get the final output value: o
        for i in range(2):
            o[i] = 1 / (1 + exp(-net4[i]))

        # -------------反向传播算法，计算各层delta值（误差E对各层权值的导数）-----------------
        # 最后一层delta值
        # 计算公式
        for i in range(2):
            delta3[i] = np.dot((y[i] - o[i]), (np.dot(o[i], (1 - o[i]))))

        # # -----第二个隐含层---
        # # 计算公式，与其后一层的delta值相关
        for j in range(3):
            delta2[j] = dot(hidden2[j], dot((1 - hidden2[j]), dot(delta3, u[:, j])))
        #
        # # -----第一个隐含层---
        # # 计算公式，与其后一层的delta值相关
        for k in range(2):
            delta1[k] = dot(hidden1[k], (1 - dot(hidden1[k], dot(delta2, v[:, k]))))
        # --------各层delta计算完后开始更新权值 - --------------------
        # ---更新u权值-----
        # 计算公式： w = w + eta * delta * x
        for i in range(2):
            for j in range(3):
                u[i, j] = u[i, j] + eta * delta3[i] * hidden2[j]

        # ---更新v权值-----
        for i in range(3):
            for j in range(2):
                v[i, j] = v[i, j] + eta * delta2[i] * hidden1[j]

        # ---更新w权值-----
        for i in range(2):
            for j in range(3):
                w[i, j] = w[i, j] + eta * delta1[i] * x[j]

        # --------------记录一下这个过程后的误差值
        # 计算误差向量  （计算输出-目标输出）
        e = np.array(o).transpose() - y

        # 算误差平方和
        sigma.append(np.dot(e, e))


BP_BACK(data, eta)
plt.plot(sigma)
plt.show()
