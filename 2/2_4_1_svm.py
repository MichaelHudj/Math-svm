# 2_4_1_svm
# ! python2.7
# referenced from https://blog.csdn.net/weixin_39257042/article/details/80402945

# dependency: numpy cvxpy
# install cvxpy: pip install pip

from cvxpy import *
import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt

# import data from q1x.dat(includes 99 samples)
q1x = np.loadtxt("q1x.dat")
# import data from q1y.dat(includes 99 classified results about samples in q1x.dat)
q1y = np.loadtxt("q1y.dat")

# initialize SVM known variables
X = q1x
y = 2 * (q1y - 0.5)
C = 1
m = X.shape[0]
n = X.shape[1]

# initialize SVM unknown variables
w = cp.Variable(n, 1)
b = cp.Variable()
xi = cp.Variable(m, 1)

# cvxpy
objective = cp.Minimize(1 / 2 * cp.norm(w) + C * sum(xi))
constraints = [cp.multiply(y, (X * w + b)) >= 1 - xi, xi >= 0]
prob = cp.Problem(objective, constraints)
result = prob.solve()
w.value

xp = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
yp = - (w.value[0] * xp + b) / w.value[1]

# margin boundary for support vectors for y=1
yp1 = - (w.value[0] * xp + b - 1) / w.value[1]
# margin boundary for support vectors for y=0
yp0 = - (w.value[0] * xp + b + 1) / w.value[1]

idx0 = np.where(q1y == 0)
idx1 = np.where(q1y == 1)
# plot negative samples
plt.plot(q1x[idx0, 0], q1x[idx0, 1], 'rx')
# plot positive samples
plt.plot(q1x[idx1, 0], q1x[idx1, 1], 'go')
# plot support vector machine
plt.plot(xp, yp.value, 'b--')
plt.plot(xp, yp1.value, 'r--')
plt.plot(xp, yp0.value, 'g--')
plt.title('decision boundary for a linear SVM classifier with C=1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
