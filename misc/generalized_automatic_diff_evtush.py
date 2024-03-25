import numpy as np
import scipy.linalg as sp
import matplotlib.pyplot as plt
func = lambda x:  x*(x - 1)

def calculate_Matrix(n):
    zerosMatrix = np.zeros([n, n], dtype=float)
    FD2_Matrix = zerosMatrix + np.diag(np.ones(n - 1), k=1)
    FD2_Matrix += np.transpose(FD2_Matrix)
    FD2_Matrix += -2*np.diag(np.ones(n))
    FD2_Matrix[0, :] *= 0
    FD2_Matrix[-1, :] *= 0
    FD2_Matrix[0, 0] = 1.0
    FD2_Matrix[-1, -1] = 1.0
    return FD2_Matrix

# def calculate_SimpsonMatrix(n):
#     FD2_Matrix_left = (-6 * np.eye(n, dtype=float) +
#                        2 * np.diag(np.ones(n - 1), k=1) +
#                        4 * np.diag(np.ones(n - 1), k=-1))
#     FD2_Matrix_right = (-6*np.eye(n, dtype=float) +
#                        4 * np.diag(np.ones(n - 1), k=1) +
#                        2 * np.diag(np.ones(n - 1), k=-1))
#
#     FD2_Matrix_left[-2:, :] = 0
#     FD2_Matrix_left[:2, :] = 0
#     FD2_Matrix_left[3::2, :] = 0
#
#     FD2_Matrix_right[:2, :] = 0
#     # FD2_Matrix_right[-2:, :] = 0
#     # FD2_Matrix_right[2::2, :] = 0
#     # print("left")
#     # print(FD2_Matrix_left)
#     # print("right")
#     # print(FD2_Matrix_right)
#     FD2_Matrix = FD2_Matrix_left + FD2_Matrix_right
#     FD2_Matrix[1, :3] = [1.0, -5.0, 4.0]
#     # FD2_Matrix[-2, -3:] = [2.0, -7.0, 5.0]
#     # FD2_Matrix[-2, -3:] *= -1
#     print(FD2_Matrix)
#     FD2_Matrix[0, 0] = 1.0
#     FD2_Matrix[-1, -1] = 1.0
#     return FD2_Matrix

for N in range(2, 20):
    # N = 5
    h = 1/N
    points = np.arange(0, N + 1)*h
    # print(points)
    matrix = calculate_Matrix(N + 1)
    # matrix = calculate_SimpsonMatrix(N + 1)
    l = -100
    simpsonsF = np.ones(N + 1) * l / 2 * h**2
    # simpsonsF = 3 * np.ones(N + 1) * l / 2 * (h)**2
    simpsonsF[0] = 0
    simpsonsF[-1] = 0
    # sol = sp.solve(matrix[1: -1, 1:-1], -f[1: -1])
    sol = sp.solve(matrix, -simpsonsF)
    print(np.max(np.abs(sol - 25*func(points))))
# plt.plot(points, sol - 25*func(points))
# plt.plot(points, 25*func(points))
# plt.show()
# reducedPoints = points[1:-1]
# plt.plot(reducedPoints, sol)
# plt.plot(reducedPoints, sol - 25*func(reducedPoints))
# plt.show()
# print(sol - func(points[1: -1]) * 25)

# evalFunc = func(points)
# diff = np.diff(evalFunc)
# print(evalFunc.shape)
# print(diff.shape)
# # plt.plot(points, evalFunc)
# # # plt.show()
# print(np.sum(diff**2)/h - 1/3)

# evtushenkoSimpsonsCoeffs = np.ones(N)*1/3
# evtushenkoSimpsonsCoeffs[1::2] *= 4
# evtushenkoSimpsonsCoeffs[2:-1:2] *= 2
# print(np.dot(evalFunc, evtushenkoSimpsonsCoeffs) * h - 625/3)



# correctSimpsonsCoeffs = np.ones(N)*1/3
# correctSimpsonsCoeffs[1::2] *= 4
# correctSimpsonsCoeffs[2:-1:2] *= 2
# print(np.dot(evalFunc, evtushenkoSimpsonsCoeffs) * h)