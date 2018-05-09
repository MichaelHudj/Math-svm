# 1_2_2_gradient_method_quadratic
# ! python3

from numpy import *
import numpy as np

A = mat([[1, 0], [0, 2]])
b = mat([[0], [0]])
x0 = mat([[2], [1]])
epsilon = 1e-5
print("A=", A)
print("b=", b)
print("x0=", x0)
print("epsilon=", epsilon)


# function [x,fun_val]=gradient_method_quadratic(A,b,x0,epsilon);
# INPUT
# ======================
# A ....... the positive definite matrix associated with the objective function
# b ....... a column vector associated with the linear part of the objective function
# x0 ...... starting point of the method
# epsilon . tolerance parameter
# OUTPUT
# =======================
# x ....... an optimal solution (up to a tolerance) of min(x^T A x+2 b^T x)
# fun_val . the optimal function value up to a tolerance

def gradient_method_quadratic(A, b, x0, epsilon):
    # x赋初值
    x = x0

    # 迭代次数赋初值
    iterCnt = 0

    # 求梯度
    grad = 2 * (A * x + b)

    # 梯度下降。停止条件：搜索方向（负梯度方向）的二范数<=epsilon
    while np.linalg.norm(grad, 2) > epsilon:
        iterCnt = iterCnt + 1

        # 步长迭代计算：a=(d^T*f'(x))/(2d^T*A*d)=(grad**2)/(2*gread.T*A*grad)
        t = (np.linalg.norm(grad, 2) ** 2) / (2 * np.linalg.norm(grad.T * A * grad, 2))

        # x(k+1) = x(k)+ad = x - t*grad
        x = x - t * grad

        # 重新计算搜索方向d = f'(x) = grad
        grad = 2 * (A * x + b)

        # 重新计算输出值y
        fun_val = x.T * A * x + 2 * b.T * x

        # 输出结果
        print("iterCnt_number = %3d norm_grad = %2.6f fun_val = %2.6f" % (iterCnt, np.linalg.norm(grad, 2), fun_val))

gradient_method_quadratic(A, b, x0, epsilon)
