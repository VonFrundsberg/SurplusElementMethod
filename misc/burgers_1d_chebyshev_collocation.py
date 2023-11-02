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
    x = np.linspace(-1, 1, 1000)
    # Create the plot with different line types
    plt.figure(figsize=(10, 6))

    plt.plot(x, asol(x, 0), label=f't={0}', linestyle='--', linewidth=2)
    plt.plot(x, asol(x, 0.65), label=f't={0.65}', linestyle='-.', linewidth=2)
    plt.plot(x, asol(x, 1), label=f't={1}', linestyle='-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.title('Solution of Burgers\' Equation')
    plt.legend()

    # Show the plot
    plt.show()
    # print('plots')
    u0_func = lambda x: asol(x, 0)
    uLeft = lambda t: asol(-1, t)
    uRight = lambda t: asol(1, t)
    uPrev = u0_func(chebPoints)
    def linearCN_nonlinearEuler(uPrev):
        curT = 0
        operatorL = v*D2
        operatorG = lambda x: x * (D @ x)
        iterMatrix = np.eye(n) - 0.5*dt*operatorL
        iterMatrix[0, :] = 0
        iterMatrix[0, 0] = 1
        iterMatrix[-1, :] = 0
        iterMatrix[-1, -1] = 1
        # t = time.time()
        invIterMatrix = sp_lin.inv(iterMatrix)
        # print(time.time() - t)
        # t = time.time()
        for i in range(iterNumber):
            rhs = uPrev - dt*(operatorG(uPrev) - 0.5 * operatorL @ uPrev)
            rhs[0] = uLeft(curT + dt)
            rhs[-1] = uRight(curT + dt)
            # uNext = sp_lin.solve(iterMatrix, rhs)
            uNext = invIterMatrix @ rhs
            curT += dt
            uPrev = uNext
        spectralSolutionError = asol(chebPoints, curT) - uNext
        return spectralSolutionError
    def bothRK4(uPrev):
        curT = 0
        operatorL = v*D2
        operatorG = lambda x: x * (D @ x)
        f = lambda x: -operatorG(x) + operatorL @ x

        for i in range(iterNumber):
            k1 = f(uPrev)
            k2 = f(uPrev + 0.5 * dt * k1)
            k3 = f(uPrev + 0.5 * dt * k2)
            k4 = f(uPrev + dt * k3)

            uNext = uPrev + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)
            uNext[0] = uLeft(curT + dt)
            uNext[-1] = uRight(curT + dt)
            curT += dt
            uPrev = uNext

        spectralSolutionError = asol(chebPoints, curT) - uNext
        return spectralSolutionError
    def linearImplicitEuler_nonlinearEuler(uPrev):
        curT = 0
        operatorL = v*D2
        operatorG = lambda x: x * (D @ x)
        iterMatrix = np.eye(n) - dt*operatorL
        iterMatrix[0, :] = 0
        iterMatrix[0, 0] = 1
        iterMatrix[-1, :] = 0
        iterMatrix[-1, -1] = 1
        # t = time.time()
        invIterMatrix = sp_lin.inv(iterMatrix)
        # print(time.time() - t)
        # t = time.time()
        for i in range(iterNumber):
            rhs = uPrev - dt*(operatorG(uPrev))
            rhs[0] = uLeft(curT + dt)
            rhs[-1] = uRight(curT + dt)
            # uNext = sp_lin.solve(iterMatrix, rhs)
            uNext = invIterMatrix @ rhs
            curT += dt
            uPrev = uNext
        spectralSolutionError = asol(chebPoints, curT) - uNext
        return spectralSolutionError
    spectralSolutionError = bothRK4(uPrev)


    def bothRK4_FD2():
        h = 2 / (n - 1)
        nodes = np.linspace(-1, 1, n)
        uPrev = u0_func(nodes)
        D = calculateFD2_Matrix(n) / h
        D2 = D @ D
        curT = 0
        operatorL = v*D2
        operatorG = lambda x: x * (D @ x)
        f = lambda x: -operatorG(x) + operatorL @ x

        for i in range(iterNumber):
            k1 = f(uPrev)
            k2 = f(uPrev + 0.5 * dt * k1)
            k3 = f(uPrev + 0.5 * dt * k2)
            k4 = f(uPrev + dt * k3)

            uNext = uPrev + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)
            uNext[0] = uLeft(curT + dt)
            uNext[-1] = uRight(curT + dt)
            curT += dt
            uPrev = uNext

        return asol(nodes, curT) - uNext

    FD2SolutionError = bothRK4_FD2()

    def bothRK4_FD4():
        h = 2 / (n - 1)
        nodes = np.linspace(-1, 1, n)
        uPrev = u0_func(nodes)
        D = calculateFD4_Matrix(n) / h
        D2 = D @ D
        curT = 0
        operatorL = v*D2
        operatorG = lambda x: x * (D @ x)
        f = lambda x: -operatorG(x) + operatorL @ x

        for i in range(iterNumber):
            k1 = f(uPrev)
            # k1[0] = uLeft(curT)
            # k1[-1] = uRight(curT)

            k2 = f(uPrev + 0.5 * dt * k1)
            # k2[0] = uLeft(curT + 0.5 * dt)
            # k2[-1] = uRight(curT + 0.5 * dt)
            k3 = f(uPrev + 0.5 * dt * k2)
            # k3[0] = uLeft(curT + 0.5 * dt)
            # k3[-1] = uRight(curT + 0.5 * dt)
            k4 = f(uPrev + dt * k3)
            # k4[0] = uLeft(curT + dt)
            # k4[-1] = uRight(curT+ dt)

            uNext = uPrev + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)
            uNext[0] = uLeft(curT + dt)
            uNext[-1] = uRight(curT + dt)
            curT += dt
            uPrev = uNext

        return asol(nodes, curT) - uNext
    FD4SolutionError = bothRK4_FD4()
    return list(map(lambda x: np.max(np.abs(x)), [spectralSolutionError, FD2SolutionError, FD4SolutionError]))
    # iterMatrix = np.eye(n) - dt*v*D2
    # iterMatrix[0, :] = 0
    # iterMatrix[0, 0] = 1
    # iterMatrix[-1, :] = 0
    # iterMatrix[-1, -1] = 1
    # invIterMatrix = sp_lin.inv(iterMatrix)
    # for i in range(iterNumber):
    #     rhs = uPrev - dt * uPrev * (D @ uPrev)
    #     rhs[0] = uLeft(curT + dt)
    #     rhs[-1] = uRight(curT + dt)
    #     # uNext = sp_lin.solve(iterMatrix, rhs)
    #     uNext = invIterMatrix @ rhs
    #     curT += dt
    #     uPrev = uNext
    #
    # FD2SolutionError = asol(nodes, curT) - uNext

    # print(time.time() - t)
    # print(spectralSolutionError)

    # h = 2 / (n - 1)
    # nodes = np.linspace(-1, 1, n)
    # uPrev = u0_func(nodes)
    # curT = 0
    # D = calculateFD2_Matrix(n)/h
    # D2 = D @ D
    # iterMatrix = np.eye(n) - dt*v*D2
    # iterMatrix[0, :] = 0
    # iterMatrix[0, 0] = 1
    # iterMatrix[-1, :] = 0
    # iterMatrix[-1, -1] = 1
    # invIterMatrix = sp_lin.inv(iterMatrix)
    # for i in range(iterNumber):
    #     rhs = uPrev - dt * uPrev * (D @ uPrev)
    #     rhs[0] = uLeft(curT + dt)
    #     rhs[-1] = uRight(curT + dt)
    #     # uNext = sp_lin.solve(iterMatrix, rhs)
    #     uNext = invIterMatrix @ rhs
    #     curT += dt
    #     uPrev = uNext
    #
    # FD2SolutionError = asol(nodes, curT) - uNext
    #
    # uPrev = u0_func(nodes)
    # curT = 0
    # D = calculateFD4_Matrix(n)/h
    # D2 = D @ D
    # iterMatrix = np.eye(n) - dt*v*D2
    # iterMatrix[0, :] = 0
    # iterMatrix[0, 0] = 1
    # iterMatrix[-1, :] = 0
    # iterMatrix[-1, -1] = 1
    # invIterMatrix = sp_lin.inv(iterMatrix)
    # for i in range(iterNumber):
    #     rhs = uPrev - dt * uPrev * (D @ uPrev)
    #     rhs[0] = uLeft(curT + dt)
    #     rhs[-1] = uRight(curT + dt)
    #     # uNext = sp_lin.solve(iterMatrix, rhs)
    #     uNext = invIterMatrix @ rhs
    #     curT += dt
    #     uPrev = uNext
    # FD4SolutionError = asol(nodes, curT) - uNext
    # # plt.plot(chebPoints, spectralSolutionError)
    # # plt.plot(nodes, FD2SolutionError)
    # # plt.plot(nodes, FD4SolutionError)
    # # plt.show()
    # print(np.max(np.abs(spectralSolutionError)))

    # # print(list(map(lambda x: np.max(np.abs(x)), [spectralSolutionError, FD2SolutionError, FD4SolutionError])))
    # return list(map(lambda x: np.max(np.abs(x)), [spectralSolutionError, FD2SolutionError, FD4SolutionError]))

errors = []
indices = []
for i in range(80, 101):
    indices.append(i)
    error = calculateBurgers(i, tFinal=1)
    errors.append(error)
    print(i, *error)
#
# errors = np.array(errors)
# indices = np.array(indices)
# np.savetxt("chebyshevColAuto.txt", np.hstack([indices[:, np.newaxis], errors]))
# data = np.loadtxt("chebyshevColAuto.txt")
data = np.loadtxt("chebyshevCol.txt")
# print(data.shape)
# plt.loglog(data[:, 0], data[:, 1:])
# plt.show()
grid_points = data[:, 0]

# Extract the error values for each method
errors_2nd_order = data[:, 1]
errors_4th_order = data[:, 2]
errors_collocation = data[:, 3]

# Create the log-log plot with different line styles
plt.figure(figsize=(8, 6))
plt.loglog(grid_points, errors_2nd_order, label='Collocation diff matrix', linestyle='-', linewidth=2)
plt.loglog(grid_points, errors_4th_order, label='2nd Order FD matrix', linestyle='--', linewidth=2)
plt.loglog(grid_points, errors_collocation, label='4th Order FD matrix', linestyle='-.', linewidth=2)

# Customize the plot
plt.xlabel('Amount of Grid Points', fontsize=14)
plt.ylabel('Approximation Error', fontsize=14)
plt.title('Approximation Errors for Burgers\' equation' , fontsize=16)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add a legend
plt.legend()

# You can also add a grid for better readability

# Show the plot
plt.show()


