import numpy as np
import mathematics.spectral as spec
import scipy.linalg as sp_lin
import matplotlib.pyplot as plt
def solvePeriodicODE_TYPE0_COLLOCATION(polyOrder, rhsF):
    """Numerically solves the equation
        du = rhsF(x), where rhsF is a periodic function, on the interval [0, 2*pi]
        Numerical method is a spectral collocation using periodic cardinal functions from Boyd (2001)"""
    n = polyOrder
    D = spec.periodicDiffMatrix(n)
    x = spec.periodicNodes(n)
    sol = sp_lin.solve(D, rhsF(x))
    return sol
def solvePeriodicODE_TYPE1_COLLOCATION(polyOrder, rhsF):
    """Numerically solves the equation
        du = rhsF(x), where rhsF is a periodic function, on the interval [0, 2*pi]
        Numerical method is a spectral collocation using periodic cardinal functions from Boyd (2001)"""
    n = polyOrder
    D = spec.periodicDiffMatrix(n)
    I = np.eye(n)
    x = spec.periodicNodes(n)
    sol = sp_lin.solve(D + I, rhsF(x))
    return sol


def solveODE_WithDifferentMethods(nodesAmount: int, func, funcDerivative):
    n = nodesAmount
    zerosMatrix = np.zeros([n, n], dtype=float)
    FD2_Matrix = zerosMatrix + 0.5 * np.diag(np.ones(n - 1), k=1)
    FD2_Matrix[0, -1] = -0.5
    FD2_Matrix -= np.transpose(FD2_Matrix)

    FD4_Matrix = (zerosMatrix +
                  2.0 / 3.0 * np.diag(np.ones(n - 1), k=1) -
                  1.0 / 12.0 * np.diag(np.ones(n - 2), k=2))

    FD4_Matrix[0, -1] = -2.0 / 3.0
    FD4_Matrix[0, -2] = 1.0 / 12.0
    FD4_Matrix[1, -1] = 1.0 / 12.0
    FD4_Matrix -= np.transpose(FD4_Matrix)

    FD8_Matrix = (zerosMatrix +
                  4.0 / 5.0 * np.diag(np.ones(n - 1), k=1) +
                  -1.0 / 5.0 * np.diag(np.ones(n - 2), k=2) +
                  4.0 / 105.0 * np.diag(np.ones(n - 3), k=3) +
                  -1.0 / 280.0 * np.diag(np.ones(n - 4), k=4))

    for i in range(4):
        FD8_Matrix[i, -4 + i:] = -FD8_Matrix[i, 2 * i + 1: 5 + i][::-1]
    FD8_Matrix -= np.transpose(FD8_Matrix)

    FD16_Matrix = (zerosMatrix +
                   8.0 / 9.0 * np.diag(np.ones(n - 1), k=1) +
                   -14.0 / 45.0 * np.diag(np.ones(n - 2), k=2) +
                   56.0 / 495.0 * np.diag(np.ones(n - 3), k=3) +
                   -7.0 / 198.0 * np.diag(np.ones(n - 4), k=4) +
                   +56.0 / 6435.0 * np.diag(np.ones(n - 5), k=5) +
                   -2.0 / 1287.0 * np.diag(np.ones(n - 6), k=6) +
                   8.0 / 45045.0 * np.diag(np.ones(n - 7), k=7) +
                   -1.0 / 102960.0 * np.diag(np.ones(n - 8), k=8))

    for i in range(8):
        FD16_Matrix[i, -8 + i:] = -FD16_Matrix[i, 2 * i + 1: 8 + 1 + i][::-1]
    FD16_Matrix -= np.transpose(FD16_Matrix)

    h = 2 * np.pi / n
    nodes = np.linspace(-np.pi + h, np.pi, n)
    funcDerivativeValuesAtNodes = funcDerivative(nodes)
    derivative2_ApproxSolution = sp_lin.solve(FD2_Matrix + h * np.eye(n), funcDerivativeValuesAtNodes * h)
    derivative4_ApproxSolution = sp_lin.solve(FD4_Matrix + h * np.eye(n), funcDerivativeValuesAtNodes * h)
    derivative8_ApproxSolution = sp_lin.solve(FD8_Matrix + h * np.eye(n), funcDerivativeValuesAtNodes * h)
    derivative16_ApproxSolution = sp_lin.solve(FD16_Matrix + h * np.eye(n), funcDerivativeValuesAtNodes * h)
    funcValuesAtNodes = func(nodes)
    FD2_ERROR = np.max(np.abs(derivative2_ApproxSolution - funcValuesAtNodes))
    FD4_ERROR = np.max(np.abs(derivative4_ApproxSolution - funcValuesAtNodes))
    FD8_ERROR = np.max(np.abs(derivative8_ApproxSolution - funcValuesAtNodes))
    FD16_ERROR = np.max(np.abs(derivative16_ApproxSolution - funcValuesAtNodes))

    D = spec.periodicDiffMatrix(n, halfInterval=False)
    np.set_printoptions(precision=3, suppress=True)
    # print(D)
    I = np.eye(n)
    x = spec.periodicNodes(n, halfInterval=False)
    # print(x - np.pi)
    sol = sp_lin.solve(D + I, funcDerivative(x - np.pi))
    # SPECTRAL_ERROR = np.max(np.abs((D + I) @ func(x - np.pi) - funcDerivative(x - np.pi)))
    SPECTRAL_ERROR = np.max(np.abs(sol - func(x - np.pi)))
    # print(SPECTRAL_ERROR)
    # plt.plot(x, sol)
    # plt.plot(x, func(x))
    # plt.show()
    # return SPECTRAL_ERROR
    return np.array([FD2_ERROR, FD4_ERROR, FD8_ERROR, FD16_ERROR, SPECTRAL_ERROR], dtype=float)
def compareODE_Solutions(nodesAmount: int, func, funcDerivative):
    n = nodesAmount
    zerosMatrix = np.zeros([n, n], dtype=float)
    FD2_Matrix = zerosMatrix + 0.5 * np.diag(np.ones(n - 1), k=1)
    FD2_Matrix[0, -1] = -0.5
    FD2_Matrix -= np.transpose(FD2_Matrix)

    FD4_Matrix = (zerosMatrix +
                  2.0 / 3.0 * np.diag(np.ones(n - 1), k=1) -
                  1.0 / 12.0 * np.diag(np.ones(n - 2), k=2))

    FD4_Matrix[0, -1] = -2.0 / 3.0
    FD4_Matrix[0, -2] = 1.0 / 12.0
    FD4_Matrix[1, -1] = 1.0 / 12.0
    FD4_Matrix -= np.transpose(FD4_Matrix)

    FD8_Matrix = (zerosMatrix +
                  4.0 / 5.0 * np.diag(np.ones(n - 1), k=1) +
                  -1.0 / 5.0 * np.diag(np.ones(n - 2), k=2) +
                  4.0 / 105.0 * np.diag(np.ones(n - 3), k=3) +
                  -1.0 / 280.0 * np.diag(np.ones(n - 4), k=4))

    for i in range(4):
        FD8_Matrix[i, -4 + i:] = -FD8_Matrix[i, 2 * i + 1: 5 + i][::-1]
    FD8_Matrix -= np.transpose(FD8_Matrix)

    FD16_Matrix = (zerosMatrix +
                   8.0 / 9.0 * np.diag(np.ones(n - 1), k=1) +
                   -14.0 / 45.0 * np.diag(np.ones(n - 2), k=2) +
                   56.0 / 495.0 * np.diag(np.ones(n - 3), k=3) +
                   -7.0 / 198.0 * np.diag(np.ones(n - 4), k=4) +
                   +56.0 / 6435.0 * np.diag(np.ones(n - 5), k=5) +
                   -2.0 / 1287.0 * np.diag(np.ones(n - 6), k=6) +
                   8.0 / 45045.0 * np.diag(np.ones(n - 7), k=7) +
                   -1.0 / 102960.0 * np.diag(np.ones(n - 8), k=8))

    for i in range(8):
        FD16_Matrix[i, -8 + i:] = -FD16_Matrix[i, 2 * i + 1: 8 + 1 + i][::-1]
    FD16_Matrix -= np.transpose(FD16_Matrix)

    h = 2 * np.pi / n
    nodes = np.linspace(-np.pi + h, np.pi, n)
    funcDerivativeValuesAtNodes = funcDerivative(nodes)
    derivative2_ApproxSolution = sp_lin.solve(FD2_Matrix + h*np.eye(n), funcDerivativeValuesAtNodes*h)
    derivative4_ApproxSolution = sp_lin.solve(FD4_Matrix + h*np.eye(n), funcDerivativeValuesAtNodes*h)
    derivative8_ApproxSolution = sp_lin.solve(FD8_Matrix + h*np.eye(n), funcDerivativeValuesAtNodes*h)
    derivative16_ApproxSolution = sp_lin.solve(FD16_Matrix + h*np.eye(n), funcDerivativeValuesAtNodes*h)
    funcValuesAtNodes = func(nodes)
    plt.plot(nodes, derivative2_ApproxSolution - funcValuesAtNodes, label="2nd order FD Approximation", color='blue',
              marker='o', linestyle='-', linewidth=2)
    plt.plot(nodes, derivative4_ApproxSolution - funcValuesAtNodes , label="4th order FD Approximation", color='green',
              marker='s', linestyle='--', linewidth=2)
    plt.plot(nodes, derivative8_ApproxSolution - funcValuesAtNodes, label="8th order FD Approximation", color='cyan',
             marker='v', linestyle=':', linewidth=2)
    plt.plot(nodes, derivative16_ApproxSolution - funcValuesAtNodes, label="16th order FD Approximation", color='purple',
             marker='x', linestyle='-.', linewidth=2)

    # Add labels, legend, and other plot enhancements
    plt.xlabel("x values")
    plt.ylabel("error values")
    plt.title("Derivative Approximation errors")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # Improves spacing
    plt.show()
def compareDerivativeCalculation(nodesAmount: int, func, funcDerivative):
    n = nodesAmount
    zerosMatrix = np.zeros([n, n], dtype=float)
    FD2_Matrix = zerosMatrix + 0.5*np.diag(np.ones(n - 1), k=1)
    FD2_Matrix[0, -1] = -0.5
    FD2_Matrix -= np.transpose(FD2_Matrix)

    FD4_Matrix = (zerosMatrix +
                  2.0/3.0 * np.diag(np.ones(n - 1), k=1) -
                  1.0/12.0 * np.diag(np.ones(n - 2), k=2))

    FD4_Matrix[0, -1] = -2.0 / 3.0
    FD4_Matrix[0, -2] = 1.0/12.0
    FD4_Matrix[1, -1] = 1.0/12.0
    FD4_Matrix -= np.transpose(FD4_Matrix)

    FD8_Matrix = (zerosMatrix +
                  4.0 / 5.0 * np.diag(np.ones(n - 1), k=1) +
                  -1.0 / 5.0 * np.diag(np.ones(n - 2), k=2) +
                  4.0 / 105.0 * np.diag(np.ones(n - 3), k=3) +
                  -1.0 / 280.0 * np.diag(np.ones(n - 4), k=4))

    for i in range(4):
        FD8_Matrix[i, -4 + i:] = -FD8_Matrix[i, 2*i + 1: 5 + i][::-1]
    FD8_Matrix -= np.transpose(FD8_Matrix)

    FD16_Matrix = (zerosMatrix +
                  8.0 / 9.0 * np.diag(np.ones(n - 1), k=1) +
                  -14.0 / 45.0 * np.diag(np.ones(n - 2), k=2) +
                  56.0 / 495.0 * np.diag(np.ones(n - 3), k=3) +
                  -7.0 / 198.0 * np.diag(np.ones(n - 4), k=4) +
                  +56.0 / 6435.0 * np.diag(np.ones(n - 5), k=5) +
                   -2.0 / 1287.0 * np.diag(np.ones(n - 6), k=6) +
                   8.0 / 45045.0 * np.diag(np.ones(n - 7), k=7) +
                   -1.0 / 102960.0 * np.diag(np.ones(n - 8), k=8))

    for i in range(8):
        FD16_Matrix[i, -8 + i:] = -FD16_Matrix[i, 2 * i + 1: 8 + 1 + i][::-1]
    FD16_Matrix -= np.transpose(FD16_Matrix)

    # print(FD2_Matrix)
    # print(FD4_Matrix)
    # print(FD8_Matrix)

    h = 2 * np.pi / n
    nodes = np.linspace(-np.pi + h, np.pi, n)
    funcValuesAtNodes = func(nodes)
    derivative2_Approx = FD2_Matrix @ funcValuesAtNodes/h
    derivative4_Approx = FD4_Matrix @ funcValuesAtNodes/h
    derivative8_Approx = FD8_Matrix @ funcValuesAtNodes / h
    derivative16_Approx = FD16_Matrix @ funcValuesAtNodes / h
    # plt.scatter(nodes, derivative2_Approx, label="2nd order FD Approximation",  color='blue',
    #          marker='o')
    # plt.scatter(nodes, derivative4_Approx, label="4th order FD Approximation",  color='green',
    #          marker='s')
    # plt.scatter(nodes, derivative8_Approx, label="8th order FD Approximation", color='cyan',
    #             marker='v')
    # plt.scatter(nodes, derivative16_Approx, label="16th order FD Approximation", color='purple',
    #             marker='x')
    # plotVals = np.linspace(-np.pi + h, np.pi, 1000)
    # plt.plot(plotVals, funcDerivative(plotVals), label="Exact Derivative", linestyle='-', color='red', linewidth=2)
    #
    # plt.xlabel("Nodes")
    # plt.ylabel("Values")
    # plt.title("Comparison of Derivative Approximations")
    #
    # plt.grid(True)
    # plt.legend(loc='best')
    # plt.show()
    evaluatedFuncDerivative = funcDerivative(nodes)
    # plt.plot(nodes, derivative2_Approx - evaluatedFuncDerivative, label="2nd order FD Approximation", color='blue',
    #          marker='o', linestyle='-', linewidth=2)
    # plt.plot(nodes, derivative4_Approx - evaluatedFuncDerivative, label="4th order FD Approximation", color='green',
    #          marker='s', linestyle='--', linewidth=2)
    plt.plot(nodes, derivative8_Approx - evaluatedFuncDerivative, label="8th order FD Approximation", color='cyan',
             marker='v', linestyle=':', linewidth=2)
    plt.plot(nodes, derivative16_Approx - evaluatedFuncDerivative, label="16th order FD Approximation", color='purple',
             marker='x', linestyle='-.', linewidth=2)

    # Add labels, legend, and other plot enhancements
    plt.xlabel("x values")
    plt.ylabel("error values")
    plt.title("Derivative Approximation errors")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # Improves spacing
    plt.show()
def solvePrintPlotPeriodicODE(polyorder, rhsF, asol):
    sol = solvePeriodicODE_TYPE0_COLLOCATION(polyorder, rhsF)
    x = spec.periodicNodes(polyorder)
    print('approximation order is: ', polyorder)
    print(sol - asol(x))
    plt.plot(x, sol)
    plt.plot(x, asol(x))
    plt.show()

# np.set_printoptions(precision=3, suppress=True)
# compareDerivativeCalculation(8,
#                              lambda x: np.exp(np.sin(x)),
#                              lambda x: np.cos(x)*np.exp(np.sin(x)))
# np.set_printoptions(precision=3, suppress=True)
# compareDerivativeCalculation(16,
#                              lambda x: np.exp(np.sin(x)**3 + np.cos(x)),
#                              lambda x: np.exp(np.sin(x)**3 + np.cos(x))*(-1 + 3 *np.cos(x)* np.sin(x))*np.sin(x))
errors_list = []
indices_list = []
for i in range(16, 360, 2):
    # errors = solveODE_WithDifferentMethods(i,
    #                      lambda x: np.exp(np.sin(5*x)**3 + np.cos(3*x)),
    #                      lambda x: -3*np.exp(np.sin(5*x)**3 + np.cos(3*x))*(np.sin(3*x) - 5 *np.cos(5*x)* np.sin(5*x)**2) +
    #                             np.exp(np.sin(5*x)**3 + np.cos(3*x)))
    errors = solveODE_WithDifferentMethods(i,
                                           lambda x: np.exp(np.sin(5 * x) ** 3 + np.cos(3 * x)),
                                           lambda x: np.exp(np.sin(5 * x) ** 3 + np.cos(3 * x)) * (
                                                       1.0 - 3.0*np.sin(3 * x) + 15 * np.cos(5*x) * np.sin(5*x)**2))
    indices_list.append(i)
    errors_list.append(errors)
plt.loglog(indices_list, errors_list)
plt.show()
    # print(i, errors)

# compareODE_Solutions(128,
#                      lambda x: np.exp(np.sin(5*x)**3 + np.cos(3*x)),
#                      lambda x: -3*np.exp(np.sin(5*x)**3 + np.cos(3*x))*(np.sin(3*x) - 5 *np.cos(5*x)* np.sin(5*x)**2) +
#                             np.exp(np.sin(5*x)**3 + np.cos(3*x)))
# compareDerivativeCalculation(32,
#                              lambda x: np.exp(np.sin(5*x)**3 + np.cos(3*x)),
#                              lambda x: -3*np.exp(np.sin(5*x)**3 + np.cos(3*x))*(np.sin(3*x) - 5 *np.cos(5*x)* np.sin(5*x)**2))