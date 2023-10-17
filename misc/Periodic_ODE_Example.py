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

def compareDerivativeCalculation(nodesAmount: int, func, funcDerivative):
    n = nodesAmount
    zerosMatrix = np.zeros([n, n], dtype=float)
    FD2_Matrix = zerosMatrix + 0.5*np.diag(np.ones(n - 1), k=1)
    FD2_Matrix[0, -1] = -0.5
    FD2_Matrix -= np.transpose(FD2_Matrix)

    FD4_Matrix = (zerosMatrix +
                  2.0/3.0 * np.diag(np.ones(n - 1), k=1) -
                  1.0/12.0 * np.diag(np.ones(n - 2), k=2))

    FD4_Matrix[0, -2] = 1.0/12.0
    FD4_Matrix[0, -1] = -2.0/3.0
    FD4_Matrix[1, -1] = 1.0/12.0
    FD4_Matrix -= np.transpose(FD4_Matrix)
    # print(FD2_Matrix)
    # print(FD4_Matrix)

    h = 2 * np.pi / n
    nodes = np.linspace(-np.pi + h, np.pi, n)
    funcValuesAtNodes = func(nodes)
    derivative2_Approx = FD2_Matrix @ funcValuesAtNodes/h
    derivative4_Approx = FD4_Matrix @ funcValuesAtNodes/h
    plt.plot(nodes, derivative2_Approx, label="2nd order FD Approximation", linestyle='--', color='blue', linewidth=2,
             marker='o')
    plt.plot(nodes, derivative4_Approx, label="4th order FD Approximation", linestyle='-.', color='green', linewidth=2,
             marker='s')
    plotVals = np.linspace(-np.pi + h, np.pi, 1000)
    plt.plot(plotVals, funcDerivative(plotVals), label="Exact Derivative", linestyle='-', color='red', linewidth=2)

    plt.xlabel("Nodes")
    plt.ylabel("Values")
    plt.title("Comparison of Derivative Approximations")

    plt.grid(True)
    plt.legend(loc='best')
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
compareDerivativeCalculation(16,
                             lambda x: np.exp(np.sin(x)),
                             lambda x: np.cos(x)*np.exp(np.sin(x)))
