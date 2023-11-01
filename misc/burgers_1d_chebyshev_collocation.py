import time

import numpy as np
import mathematics.spectral as spec
import scipy.linalg as sp_lin
import matplotlib.pyplot as plt
import mathematics.nonlinear as nonlin

def calculateFD4_Matrix(argN):
    n = argN
    if (n < 5):
        n = 5
    zerosMatrix = np.zeros([n, n], dtype=float)
    FD4_Matrix = (zerosMatrix +
                  2.0 / 3.0 * np.diag(np.ones(n - 1), k=1) -
                  1.0 / 12.0 * np.diag(np.ones(n - 2), k=2))
    FD4_Matrix -= np.transpose(FD4_Matrix)

    FD4_Matrix[0, :5] = [-25 / 12, 4, -3, 4 / 3, -1 / 4]
    FD4_Matrix[1, :5] = [-1 / 4, -5 / 6, 3 / 2, -1 / 2, 1 / 12]

    for i in range(2):
        FD4_Matrix[-1 - i, -5:] = -FD4_Matrix[i, :5][::-1]
    return FD4_Matrix
def calculateFD2_Matrix(n):
    zerosMatrix = np.zeros([n, n], dtype=float)
    FD2_Matrix = zerosMatrix + 0.5 * np.diag(np.ones(n - 1), k=1)
    FD2_Matrix -= np.transpose(FD2_Matrix)
    FD2_Matrix[0, :3] = [-3.0 / 2.0, + 2.0, - 1.0 / 2.0]
    FD2_Matrix[-1, -3:] = -FD2_Matrix[0, :3][::-1]
    return FD2_Matrix

def calculateBurgers(n, tFinal):
    # n = 20
    chebPoints = spec.chebNodes(n, -1, 1)
    D = spec.chebDiffMatrix(n, -1, 1)
    D2 = D @ D
    dt = 1e-5
    # uPrev = np.ones(n)
    v = 0.01
    t0 = 1.0
    a = 16
    c = 1
    iterNumber = int(tFinal / dt)
    phi = lambda x, t: x/t * (np.sqrt(a/t)*np.exp(-x**2/(4*v*t)))/(1 + np.sqrt(a/t)*np.exp(-x**2/(4*v*t)))
    asol = lambda x, t: c + phi((x - c*t), t + t0)

    # plt.plot(chebPoints, asol(chebPoints, 0))
    # plt.plot(chebPoints, asol(chebPoints, 1))
    # plt.show()

    u0_func = lambda x: asol(x, 0)
    uLeft = lambda t: asol(-1, t)
    uRight = lambda t: asol(1, t)
    uPrev = u0_func(chebPoints)

    curT = 0
    iterMatrix = np.eye(n) - dt*v*D2
    iterMatrix[0, :] = 0
    iterMatrix[0, 0] = 1
    iterMatrix[-1, :] = 0
    iterMatrix[-1, -1] = 1
    # t = time.time()
    invIterMatrix = sp_lin.inv(iterMatrix)
    # print(time.time() - t)
    # t = time.time()
    for i in range(iterNumber):
        rhs = uPrev - dt * uPrev * (D @ uPrev)
        rhs[0] = uLeft(curT + dt)
        rhs[-1] = uRight(curT + dt)
        # uNext = sp_lin.solve(iterMatrix, rhs)
        uNext = invIterMatrix @ rhs
        curT += dt
        uPrev = uNext
    spectralSolutionError = asol(chebPoints, curT) - uNext
    # print(time.time() - t)
    # print(spectralSolutionError)

    h = 2 / (n - 1)
    nodes = np.linspace(-1, 1, n)
    uPrev = u0_func(nodes)
    curT = 0
    D = calculateFD2_Matrix(n)/h
    D2 = D @ D
    iterMatrix = np.eye(n) - dt*v*D2
    iterMatrix[0, :] = 0
    iterMatrix[0, 0] = 1
    iterMatrix[-1, :] = 0
    iterMatrix[-1, -1] = 1
    invIterMatrix = sp_lin.inv(iterMatrix)
    for i in range(iterNumber):
        rhs = uPrev - dt * uPrev * (D @ uPrev)
        rhs[0] = uLeft(curT + dt)
        rhs[-1] = uRight(curT + dt)
        # uNext = sp_lin.solve(iterMatrix, rhs)
        uNext = invIterMatrix @ rhs
        curT += dt
        uPrev = uNext

    FD2SolutionError = asol(nodes, curT) - uNext

    uPrev = u0_func(nodes)
    curT = 0
    D = calculateFD4_Matrix(n)/h
    D2 = D @ D
    iterMatrix = np.eye(n) - dt*v*D2
    iterMatrix[0, :] = 0
    iterMatrix[0, 0] = 1
    iterMatrix[-1, :] = 0
    iterMatrix[-1, -1] = 1
    invIterMatrix = sp_lin.inv(iterMatrix)
    for i in range(iterNumber):
        rhs = uPrev - dt * uPrev * (D @ uPrev)
        rhs[0] = uLeft(curT + dt)
        rhs[-1] = uRight(curT + dt)
        # uNext = sp_lin.solve(iterMatrix, rhs)
        uNext = invIterMatrix @ rhs
        curT += dt
        uPrev = uNext
    FD4SolutionError = asol(nodes, curT) - uNext
    # plt.plot(chebPoints, spectralSolutionError)
    # plt.plot(nodes, FD2SolutionError)
    # plt.plot(nodes, FD4SolutionError)
    # plt.show()
    # print(list(map(lambda x: np.max(np.abs(x)), [spectralSolutionError, FD2SolutionError, FD4SolutionError])))
    return list(map(lambda x: np.max(np.abs(x)), [spectralSolutionError, FD2SolutionError, FD4SolutionError]))

errors = []
indices = []
for i in range(5, 31):
    indices.append(i)
    error = calculateBurgers(i, tFinal=1)
    errors.append(error)
    print(i, *error)

errors = np.array(errors)
indices = np.array(indices)
np.savetxt("chebyshevColAuto.txt", np.hstack([indices[:, np.newaxis], errors]))
data = np.loadtxt("chebyshevColAuto.txt")
data = np.loadtxt("chebyshevCol.txt")
# print(data.shape)
plt.loglog(data[:, 0], data[:, 1:])
plt.show()

# plt.scatter(chebPoints, spectralSolutionError)
# plt.scatter(nodes, FD2SolutionError)
# plt.scatter(nodes, FD4SolutionError)
# plt.show()


