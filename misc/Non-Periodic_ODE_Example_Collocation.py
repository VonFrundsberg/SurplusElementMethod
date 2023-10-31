import numpy as np
import mathematics.spectral as spec
import scipy.linalg as sp_lin
import numpy.linalg as np_lin
import matplotlib.pyplot as plt
import time as time
from matplotlib.ticker import ScalarFormatter


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

def calculateFD8_Matrix(argN):
    n = argN
    if (n < 9):
        n = 9
    zerosMatrix = np.zeros([n, n], dtype=float)
    FD8_Matrix = (zerosMatrix +
                  4.0 / 5.0 * np.diag(np.ones(n - 1), k=1) +
                  -1.0 / 5.0 * np.diag(np.ones(n - 2), k=2) +
                  4.0 / 105.0 * np.diag(np.ones(n - 3), k=3) +
                  -1.0 / 280.0 * np.diag(np.ones(n - 4), k=4))
    FD8_Matrix -= np.transpose(FD8_Matrix)

    FD8_Matrix[0, :9] = [-761 / 280, 8, -14, 56 / 3, -35 / 2, 56 / 5, -14 / 3, 8 / 7, -1 / 8]
    FD8_Matrix[1, :9] = [-1 / 8, -223 / 140, 7 / 2, -7 / 2, 35 / 12, -7 / 4, 7 / 10, -1 / 6, 1 / 56]
    FD8_Matrix[2, :9] = [1 / 56, -2 / 7, -19 / 20, 2, -5 / 4, 2 / 3, -1 / 4, 2 / 35, -1 / 168]
    FD8_Matrix[3, :9] = [-1 / 168, 1 / 14, -1 / 2, -9 / 20, 5 / 4, -1 / 2, 1 / 6, -1 / 28, 1 / 280]

    for i in range(4):
        FD8_Matrix[-1 - i, -9:] = -FD8_Matrix[i, :9][::-1]
    return FD8_Matrix
def calculateFD16_Matrix(argN):
    n = argN
    if (n < 17):
        n = 17
    zerosMatrix = np.zeros([n, n], dtype=float)
    FD16_Matrix = (zerosMatrix +
                   8.0 / 9.0 * np.diag(np.ones(n - 1), k=1) +
                   -14.0 / 45.0 * np.diag(np.ones(n - 2), k=2) +
                   56.0 / 495.0 * np.diag(np.ones(n - 3), k=3) +
                   -7.0 / 198.0 * np.diag(np.ones(n - 4), k=4) +
                   +56.0 / 6435.0 * np.diag(np.ones(n - 5), k=5) +
                   -2.0 / 1287.0 * np.diag(np.ones(n - 6), k=6) +
                   8.0 / 45045.0 * np.diag(np.ones(n - 7), k=7) +
                   -1.0 / 102960.0 * np.diag(np.ones(n - 8), k=8))
    FD16_Matrix -= np.transpose(FD16_Matrix)
    FD16_Matrix[0, :17] = [-3.3807289932289932, 16, -60, 186.66666666666666, -455, 873.6, -1334.6666666666667,
                           1634.2857142857142, -1608.75,
                           1271.111111111111, -800.8, 397.09090909090907, -151.66666666666666, 43.07692307692308,
                           -8.571428571428571,
                           1.0666666666666667, -0.0625]
    FD16_Matrix[1, :17] = [-0.0625, -2.3182289932289932, 7.5, -17.5, 37.916666666666664, -68.25, 100.1,
                           -119.16666666666667,
                           114.91071428571429, -89.375, 55.611111111111114, -27.3, 10.340909090909092,
                           -2.9166666666666665,
                           0.5769230769230769, -0.07142857142857142, 0.004166666666666667]
    FD16_Matrix[2, :17] = [0.004166666666666667, -0.13333333333333333, -1.7515623265623266, 4.666666666666667,
                           -7.583333333333333,
                           12.133333333333333, -16.683333333333334, 19.066666666666666, -17.875, 13.619047619047619,
                           -8.341666666666667,
                           4.044444444444444, -1.5166666666666666, 0.42424242424242425, -0.08333333333333333,
                           0.010256410256410256,
                           -0.0005952380952380953]
    FD16_Matrix[3, :17] = [-0.0005952380952380953, 0.014285714285714285, -0.21428571428571427, -1.3468004218004217,
                           3.25, -3.9,
                           4.766666666666667, -5.107142857142857, 4.5964285714285715, -3.4047619047619047,
                           2.0428571428571427, -0.975,
                           0.3611111111111111, -0.1, 0.01948051948051948, -0.002380952380952381, 0.00013736263736263736]
    FD16_Matrix[4, :17] = [0.00013736263736263736, -0.0029304029304029304, 0.03296703296703297, -0.3076923076923077,
                           -1.0198773448773448, 2.4,
                           -2.2, 2.0952380952380953, -1.7678571428571428, 1.2571428571428571, -0.7333333333333333,
                           0.34285714285714286,
                           -0.125, 0.03418803418803419, -0.006593406593406593, 0.0007992007992007992,
                           -0.00004578754578754579]
    FD16_Matrix[5, :17] = [-0.00004578754578754579, 0.0009157509157509158, -0.009157509157509158, 0.0641025641025641,
                           -0.4166666666666667,
                           -0.7365440115440115, 1.8333333333333333, -1.3095238095238095, 0.9821428571428571,
                           -0.6547619047619048,
                           0.36666666666666664, -0.16666666666666666, 0.05952380952380952, -0.016025641025641024,
                           0.0030525030525030525,
                           -0.0003663003663003663, 0.000020812520812520813]
    FD16_Matrix[6, :17] = [0.000020812520812520813, -0.0003996003996003996, 0.0037462537462537465,
                           -0.023310023310023312, 0.11363636363636363,
                           -0.5454545454545454, -0.478968253968254, 1.4285714285714286, -0.8035714285714286,
                           0.47619047619047616, -0.25,
                           0.10909090909090909, -0.03787878787878788, 0.00999000999000999, -0.0018731268731268732,
                           0.000222000222000222,
                           -0.000012487512487512488]
    FD16_Matrix[7, :17] = [-0.000012487512487512488, 0.0002331002331002331, -0.002097902097902098, 0.012237762237762238,
                           -0.05303030303030303,
                           0.19090909090909092, -0.7, -0.2361111111111111, 1.125, -0.5, 0.23333333333333334,
                           -0.09545454545454546,
                           0.031818181818181815, -0.008158508158508158, 0.0014985014985014985, -0.00017482517482517483,
                           9.712509712509713e-6]
    for i in range(8):
        FD16_Matrix[-1 - i, -17:] = -FD16_Matrix[i, :17][::-1]
    return FD16_Matrix
def compareDerivativeCalculation(nodesAmount: int, func, funcDerivative):
    n = nodesAmount
    FD2_Matrix = calculateFD2_Matrix(n)
    FD4_Matrix = calculateFD2_Matrix(n)
    FD8_Matrix = calculateFD2_Matrix(n)
    FD16_Matrix = calculateFD2_Matrix(n)
    h = 2 / n
    nodes = np.linspace(-1 + h, 1, n)
    funcValuesAtNodes = func(nodes)
    derivative2_Approx = FD2_Matrix @ funcValuesAtNodes/h
    derivative4_Approx = FD4_Matrix @ funcValuesAtNodes/h
    derivative8_Approx = FD8_Matrix @ funcValuesAtNodes / h
    derivative16_Approx = FD16_Matrix @ funcValuesAtNodes / h
    plt.scatter(nodes, derivative2_Approx, label="2nd order FD Approximation",  color='blue',
             marker='o')
    plt.scatter(nodes, derivative4_Approx, label="4th order FD Approximation",  color='green',
             marker='s')
    plt.scatter(nodes, derivative8_Approx, label="8th order FD Approximation", color='cyan',
                marker='v')
    plt.scatter(nodes, derivative16_Approx, label="16th order FD Approximation", color='purple',
                marker='x')
    plotVals = np.linspace(-1 + h, 1, 1000)
    plt.plot(plotVals, funcDerivative(plotVals), label="Exact Derivative", linestyle='-', color='red', linewidth=2)
    plt.xlabel("Nodes")
    plt.ylabel("Values")
    plt.title("Comparison of Derivative Approximations")

    plt.grid(True)
    plt.legend(loc='best')
    plt.show()
    evaluatedFuncDerivative = funcDerivative(nodes)
    plt.plot(nodes, derivative2_Approx - evaluatedFuncDerivative, label="2nd order FD Approximation", color='blue',
             marker='o', linestyle='-', linewidth=2)
    plt.plot(nodes, derivative4_Approx - evaluatedFuncDerivative, label="4th order FD Approximation", color='green',
             marker='s', linestyle='--', linewidth=2)
    plt.plot(nodes, derivative8_Approx - evaluatedFuncDerivative, label="8th order FD Approximation", color='cyan',
             marker='v', linestyle=':', linewidth=2)
    plt.plot(nodes, derivative16_Approx - evaluatedFuncDerivative, label="16th order FD Approximation", color='purple',
             marker='x', linestyle='-.', linewidth=2)

    #Add labels, legend, and other plot enhancements
    plt.xlabel("x values")
    plt.ylabel("error values")
    plt.title("Derivative Approximation errors")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # Improves spacing
    plt.show()
def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)


def firstOrderODE_Zero_Dirichlet(nodesAmount: int, func, funcDerivative, func2Derivative):
    n = nodesAmount
    FD2_Matrix = calculateFD2_Matrix(n)
    FD4_Matrix = calculateFD4_Matrix(n)
    FD8_Matrix = calculateFD8_Matrix(n)
    FD16_Matrix = calculateFD16_Matrix(n)

    FD2_Matrix = (FD2_Matrix )[:-1, :-1]
    FD4_Matrix = (FD4_Matrix )[:-1, :-1]
    FD8_Matrix = (FD8_Matrix )[:-1, :-1]
    FD16_Matrix = (FD16_Matrix )[:-1, :-1]

    def solveODE(FD_matrix):
        N = np.shape(FD_matrix)[0] + 1
        h = 2 / (N - 1)
        nodes = np.linspace(-1, 1, N)[:-1]
        derivativeValues = funcDerivative(nodes)
        return sp_lin.solve(FD_matrix, derivativeValues * h)


    derivative2_ApproxSolution = solveODE(FD2_Matrix)
    derivative4_ApproxSolution = solveODE(FD4_Matrix)
    derivative8_ApproxSolution = solveODE(FD8_Matrix)
    derivative16_ApproxSolution = solveODE(FD16_Matrix)

    def calculateError(approxSolution):
        N = np.size(approxSolution) + 1
        nodes = np.linspace(-1, 1, N)[:-1]
        funcValues = func(nodes)
        return np.max(np.abs(approxSolution - funcValues))

    FD2_ERROR = calculateError(derivative2_ApproxSolution)
    FD4_ERROR = calculateError(derivative4_ApproxSolution)
    FD8_ERROR = calculateError(derivative8_ApproxSolution)
    FD16_ERROR = calculateError(derivative16_ApproxSolution)

    D = spec.chebDiffMatrix(n, -1, 1)[:-1, :-1]
    x = spec.chebNodes(n, -1, 1)[:-1]

    sol = sp_lin.solve(D, funcDerivative(x))
    SPECTRAL_ERROR = np.max(np.abs(func(x) - sol))
    maxFuncValue = np.max(np.abs(func(x)))
    nonZeroCount = np.array(list(map(lambda x: np.count_nonzero(x), [FD2_Matrix, FD4_Matrix, FD8_Matrix, FD16_Matrix, D])))
    # nonZeroCount = np.ones(5) * nodesAmount
    print(nodesAmount, SPECTRAL_ERROR, maxFuncValue)
    return [nonZeroCount,
            np.array([FD2_ERROR, FD4_ERROR, FD8_ERROR, FD16_ERROR, SPECTRAL_ERROR], dtype=float) / maxFuncValue]
def secondOrderODE_Zero_Dirichlet(nodesAmount: int, func, funcDerivative, func2Derivative):
    n = nodesAmount
    zerosMatrix = np.zeros([n, n], dtype=float)
    FD2_Matrix = zerosMatrix + 0.5 * np.diag(np.ones(n - 1), k=1)
    FD2_Matrix -= np.transpose(FD2_Matrix)
    FD2_Matrix[0, :3] = [-3.0 / 2.0, + 2.0, - 1.0 / 2.0]
    FD2_Matrix[-1, -3:] = -FD2_Matrix[0, :3][::-1]

    FD4_Matrix = (zerosMatrix +
                  2.0 / 3.0 * np.diag(np.ones(n - 1), k=1) -
                  1.0 / 12.0 * np.diag(np.ones(n - 2), k=2))
    FD4_Matrix -= np.transpose(FD4_Matrix)

    FD4_Matrix[0, :5] = [-25 / 12, 4, -3, 4 / 3, -1 / 4]
    FD4_Matrix[1, :5] = [-1 / 4, -5 / 6, 3 / 2, -1 / 2, 1 / 12]

    for i in range(2):
        FD4_Matrix[-1 - i, -5:] = -FD4_Matrix[i, :5][::-1]

    FD8_Matrix = (zerosMatrix +
                  4.0 / 5.0 * np.diag(np.ones(n - 1), k=1) +
                  -1.0 / 5.0 * np.diag(np.ones(n - 2), k=2) +
                  4.0 / 105.0 * np.diag(np.ones(n - 3), k=3) +
                  -1.0 / 280.0 * np.diag(np.ones(n - 4), k=4))
    FD8_Matrix -= np.transpose(FD8_Matrix)

    FD8_Matrix[0, :9] = [-761 / 280, 8, -14, 56 / 3, -35 / 2, 56 / 5, -14 / 3, 8 / 7, -1 / 8]
    FD8_Matrix[1, :9] = [-1 / 8, -223 / 140, 7 / 2, -7 / 2, 35 / 12, -7 / 4, 7 / 10, -1 / 6, 1 / 56]
    FD8_Matrix[2, :9] = [1 / 56, -2 / 7, -19 / 20, 2, -5 / 4, 2 / 3, -1 / 4, 2 / 35, -1 / 168]
    FD8_Matrix[3, :9] = [-1 / 168, 1 / 14, -1 / 2, -9 / 20, 5 / 4, -1 / 2, 1 / 6, -1 / 28, 1 / 280]

    for i in range(4):
        FD8_Matrix[-1 - i, -9:] = -FD8_Matrix[i, :9][::-1]

    FD16_Matrix = (zerosMatrix +
                   8.0 / 9.0 * np.diag(np.ones(n - 1), k=1) +
                   -14.0 / 45.0 * np.diag(np.ones(n - 2), k=2) +
                   56.0 / 495.0 * np.diag(np.ones(n - 3), k=3) +
                   -7.0 / 198.0 * np.diag(np.ones(n - 4), k=4) +
                   +56.0 / 6435.0 * np.diag(np.ones(n - 5), k=5) +
                   -2.0 / 1287.0 * np.diag(np.ones(n - 6), k=6) +
                   8.0 / 45045.0 * np.diag(np.ones(n - 7), k=7) +
                   -1.0 / 102960.0 * np.diag(np.ones(n - 8), k=8))
    FD16_Matrix -= np.transpose(FD16_Matrix)
    FD16_Matrix[0, :17] = [-3.3807289932289932, 16, -60, 186.66666666666666, -455, 873.6, -1334.6666666666667,
                           1634.2857142857142, -1608.75,
                           1271.111111111111, -800.8, 397.09090909090907, -151.66666666666666, 43.07692307692308,
                           -8.571428571428571,
                           1.0666666666666667, -0.0625]
    FD16_Matrix[1, :17] = [-0.0625, -2.3182289932289932, 7.5, -17.5, 37.916666666666664, -68.25, 100.1,
                           -119.16666666666667,
                           114.91071428571429, -89.375, 55.611111111111114, -27.3, 10.340909090909092,
                           -2.9166666666666665,
                           0.5769230769230769, -0.07142857142857142, 0.004166666666666667]
    FD16_Matrix[2, :17] = [0.004166666666666667, -0.13333333333333333, -1.7515623265623266, 4.666666666666667,
                           -7.583333333333333,
                           12.133333333333333, -16.683333333333334, 19.066666666666666, -17.875, 13.619047619047619,
                           -8.341666666666667,
                           4.044444444444444, -1.5166666666666666, 0.42424242424242425, -0.08333333333333333,
                           0.010256410256410256,
                           -0.0005952380952380953]
    FD16_Matrix[3, :17] = [-0.0005952380952380953, 0.014285714285714285, -0.21428571428571427, -1.3468004218004217,
                           3.25, -3.9,
                           4.766666666666667, -5.107142857142857, 4.5964285714285715, -3.4047619047619047,
                           2.0428571428571427, -0.975,
                           0.3611111111111111, -0.1, 0.01948051948051948, -0.002380952380952381, 0.00013736263736263736]
    FD16_Matrix[4, :17] = [0.00013736263736263736, -0.0029304029304029304, 0.03296703296703297, -0.3076923076923077,
                           -1.0198773448773448, 2.4,
                           -2.2, 2.0952380952380953, -1.7678571428571428, 1.2571428571428571, -0.7333333333333333,
                           0.34285714285714286,
                           -0.125, 0.03418803418803419, -0.006593406593406593, 0.0007992007992007992,
                           -0.00004578754578754579]
    FD16_Matrix[5, :17] = [-0.00004578754578754579, 0.0009157509157509158, -0.009157509157509158, 0.0641025641025641,
                           -0.4166666666666667,
                           -0.7365440115440115, 1.8333333333333333, -1.3095238095238095, 0.9821428571428571,
                           -0.6547619047619048,
                           0.36666666666666664, -0.16666666666666666, 0.05952380952380952, -0.016025641025641024,
                           0.0030525030525030525,
                           -0.0003663003663003663, 0.000020812520812520813]
    FD16_Matrix[6, :17] = [0.000020812520812520813, -0.0003996003996003996, 0.0037462537462537465,
                           -0.023310023310023312, 0.11363636363636363,
                           -0.5454545454545454, -0.478968253968254, 1.4285714285714286, -0.8035714285714286,
                           0.47619047619047616, -0.25,
                           0.10909090909090909, -0.03787878787878788, 0.00999000999000999, -0.0018731268731268732,
                           0.000222000222000222,
                           -0.000012487512487512488]
    FD16_Matrix[7, :17] = [-0.000012487512487512488, 0.0002331002331002331, -0.002097902097902098, 0.012237762237762238,
                           -0.05303030303030303,
                           0.19090909090909092, -0.7, -0.2361111111111111, 1.125, -0.5, 0.23333333333333334,
                           -0.09545454545454546,
                           0.031818181818181815, -0.008158508158508158, 0.0014985014985014985, -0.00017482517482517483,
                           9.712509712509713e-6]
    for i in range(8):
        FD16_Matrix[-1 - i, -17:] = -FD16_Matrix[i, :17][::-1]


    FD2_Matrix2 = (FD2_Matrix @ FD2_Matrix)[1:-1, 1:-1]
    FD4_Matrix2 = (FD4_Matrix @ FD4_Matrix)[1:-1, 1:-1]
    FD8_Matrix2 = (FD8_Matrix @ FD8_Matrix)[1:-1, 1:-1]
    FD16_Matrix2 = (FD16_Matrix @ FD16_Matrix)[1:-1, 1:-1]

    h = 2 / (n - 1)
    nodes = np.linspace(-1, 1, n)[1:-1]
    # print(nodes)
    secondDerivativeValues = func2Derivative(nodes)
    derivative2_ApproxSolution = sp_lin.solve(FD2_Matrix2, secondDerivativeValues * h * h)
    derivative4_ApproxSolution = sp_lin.solve(FD4_Matrix2, secondDerivativeValues * h * h)
    derivative8_ApproxSolution = sp_lin.solve(FD8_Matrix2, secondDerivativeValues * h * h)
    derivative16_ApproxSolution = sp_lin.solve(FD16_Matrix2, secondDerivativeValues * h * h)

    funcValues = func(nodes)
    FD2_ERROR = np.max(np.abs(derivative2_ApproxSolution - funcValues))
    FD4_ERROR = np.max(np.abs(derivative4_ApproxSolution - funcValues))
    FD8_ERROR = np.max(np.abs(derivative8_ApproxSolution - funcValues))
    FD16_ERROR = np.max(np.abs(derivative16_ApproxSolution - funcValues))

    D = spec.chebDiffMatrix(n, -1, 1)
    D2 = D @ D
    D2 = D2[1:-1, 1:-1]
    x = spec.chebNodes(n, -1, 1)[1:-1]
    # print(sp_lin.con)
    sol = sp_lin.solve(D2, func2Derivative(x))
    SPECTRAL_ERROR = np.max(np.abs(func(x) - sol))
    # x = np.linspace(-1, 1, 1000)
    # plt.plot(x, sol)
    # plt.plot(x, func(x))
    # plt.show()
    # plt.plot(x, func2Derivative(x))
    # plt.show()
    maxFuncValue = np.max(np.abs(func(x)))
    print(nodesAmount, SPECTRAL_ERROR, maxFuncValue)
    # return np.array([SPECTRAL_ERROR], dtype=float) / maxFuncValue
    return np.array([FD2_ERROR, FD4_ERROR, FD8_ERROR, FD16_ERROR, SPECTRAL_ERROR], dtype=float) / maxFuncValue
def derivativeError_WithDifferentMethods(nodesAmount: int, func, funcDerivative):
    n = nodesAmount
    zerosMatrix = np.zeros([n, n], dtype=float)
    FD2_Matrix = zerosMatrix + 0.5 * np.diag(np.ones(n - 1), k=1)
    FD2_Matrix -= np.transpose(FD2_Matrix)
    FD2_Matrix[0, :3] = [-3.0 / 2.0, + 2.0, - 1.0 / 2.0]
    FD2_Matrix[-1, -3:] = -FD2_Matrix[0, :3][::-1]

    FD4_Matrix = (zerosMatrix +
                  2.0 / 3.0 * np.diag(np.ones(n - 1), k=1) -
                  1.0 / 12.0 * np.diag(np.ones(n - 2), k=2))
    FD4_Matrix -= np.transpose(FD4_Matrix)

    FD4_Matrix[0, :5] = [-25 / 12, 4, -3, 4 / 3, -1 / 4]
    FD4_Matrix[1, :5] = [-1 / 4, -5 / 6, 3 / 2, -1 / 2, 1 / 12]

    for i in range(2):
        FD4_Matrix[-1 - i, -5:] = -FD4_Matrix[i, :5][::-1]

    FD8_Matrix = (zerosMatrix +
                  4.0 / 5.0 * np.diag(np.ones(n - 1), k=1) +
                  -1.0 / 5.0 * np.diag(np.ones(n - 2), k=2) +
                  4.0 / 105.0 * np.diag(np.ones(n - 3), k=3) +
                  -1.0 / 280.0 * np.diag(np.ones(n - 4), k=4))
    FD8_Matrix -= np.transpose(FD8_Matrix)

    FD8_Matrix[0, :9] = [-761 / 280, 8, -14, 56 / 3, -35 / 2, 56 / 5, -14 / 3, 8 / 7, -1 / 8]
    FD8_Matrix[1, :9] = [-1 / 8, -223 / 140, 7 / 2, -7 / 2, 35 / 12, -7 / 4, 7 / 10, -1 / 6, 1 / 56]
    FD8_Matrix[2, :9] = [1 / 56, -2 / 7, -19 / 20, 2, -5 / 4, 2 / 3, -1 / 4, 2 / 35, -1 / 168]
    FD8_Matrix[3, :9] = [-1 / 168, 1 / 14, -1 / 2, -9 / 20, 5 / 4, -1 / 2, 1 / 6, -1 / 28, 1 / 280]

    for i in range(4):
        FD8_Matrix[-1 - i, -9:] = -FD8_Matrix[i, :9][::-1]

    FD16_Matrix = (zerosMatrix +
                   8.0 / 9.0 * np.diag(np.ones(n - 1), k=1) +
                   -14.0 / 45.0 * np.diag(np.ones(n - 2), k=2) +
                   56.0 / 495.0 * np.diag(np.ones(n - 3), k=3) +
                   -7.0 / 198.0 * np.diag(np.ones(n - 4), k=4) +
                   +56.0 / 6435.0 * np.diag(np.ones(n - 5), k=5) +
                   -2.0 / 1287.0 * np.diag(np.ones(n - 6), k=6) +
                   8.0 / 45045.0 * np.diag(np.ones(n - 7), k=7) +
                   -1.0 / 102960.0 * np.diag(np.ones(n - 8), k=8))
    FD16_Matrix -= np.transpose(FD16_Matrix)
    FD16_Matrix[0, :17] = [-3.3807289932289932, 16, -60, 186.66666666666666, -455, 873.6, -1334.6666666666667,
                           1634.2857142857142, -1608.75,
                           1271.111111111111, -800.8, 397.09090909090907, -151.66666666666666, 43.07692307692308,
                           -8.571428571428571,
                           1.0666666666666667, -0.0625]
    FD16_Matrix[1, :17] = [-0.0625, -2.3182289932289932, 7.5, -17.5, 37.916666666666664, -68.25, 100.1,
                           -119.16666666666667,
                           114.91071428571429, -89.375, 55.611111111111114, -27.3, 10.340909090909092,
                           -2.9166666666666665,
                           0.5769230769230769, -0.07142857142857142, 0.004166666666666667]
    FD16_Matrix[2, :17] = [0.004166666666666667, -0.13333333333333333, -1.7515623265623266, 4.666666666666667,
                           -7.583333333333333,
                           12.133333333333333, -16.683333333333334, 19.066666666666666, -17.875, 13.619047619047619,
                           -8.341666666666667,
                           4.044444444444444, -1.5166666666666666, 0.42424242424242425, -0.08333333333333333,
                           0.010256410256410256,
                           -0.0005952380952380953]
    FD16_Matrix[3, :17] = [-0.0005952380952380953, 0.014285714285714285, -0.21428571428571427, -1.3468004218004217,
                           3.25, -3.9,
                           4.766666666666667, -5.107142857142857, 4.5964285714285715, -3.4047619047619047,
                           2.0428571428571427, -0.975,
                           0.3611111111111111, -0.1, 0.01948051948051948, -0.002380952380952381, 0.00013736263736263736]
    FD16_Matrix[4, :17] = [0.00013736263736263736, -0.0029304029304029304, 0.03296703296703297, -0.3076923076923077,
                           -1.0198773448773448, 2.4,
                           -2.2, 2.0952380952380953, -1.7678571428571428, 1.2571428571428571, -0.7333333333333333,
                           0.34285714285714286,
                           -0.125, 0.03418803418803419, -0.006593406593406593, 0.0007992007992007992,
                           -0.00004578754578754579]
    FD16_Matrix[5, :17] = [-0.00004578754578754579, 0.0009157509157509158, -0.009157509157509158, 0.0641025641025641,
                           -0.4166666666666667,
                           -0.7365440115440115, 1.8333333333333333, -1.3095238095238095, 0.9821428571428571,
                           -0.6547619047619048,
                           0.36666666666666664, -0.16666666666666666, 0.05952380952380952, -0.016025641025641024,
                           0.0030525030525030525,
                           -0.0003663003663003663, 0.000020812520812520813]
    FD16_Matrix[6, :17] = [0.000020812520812520813, -0.0003996003996003996, 0.0037462537462537465,
                           -0.023310023310023312, 0.11363636363636363,
                           -0.5454545454545454, -0.478968253968254, 1.4285714285714286, -0.8035714285714286,
                           0.47619047619047616, -0.25,
                           0.10909090909090909, -0.03787878787878788, 0.00999000999000999, -0.0018731268731268732,
                           0.000222000222000222,
                           -0.000012487512487512488]
    FD16_Matrix[7, :17] = [-0.000012487512487512488, 0.0002331002331002331, -0.002097902097902098, 0.012237762237762238,
                           -0.05303030303030303,
                           0.19090909090909092, -0.7, -0.2361111111111111, 1.125, -0.5, 0.23333333333333334,
                           -0.09545454545454546,
                           0.031818181818181815, -0.008158508158508158, 0.0014985014985014985, -0.00017482517482517483,
                           9.712509712509713e-6]
    for i in range(8):
        FD16_Matrix[-1 - i, -17:] = -FD16_Matrix[i, :17][::-1]

    h = 2 / (n - 1)
    nodes = np.linspace(-1, 1, n)
    # print(nodes)
    funcValuesAtNodes = func(nodes)
    derivative2_ApproxSolution = FD2_Matrix @ funcValuesAtNodes / h
    derivative4_ApproxSolution = FD4_Matrix @ funcValuesAtNodes / h
    derivative8_ApproxSolution = FD8_Matrix @ funcValuesAtNodes / h
    derivative16_ApproxSolution = FD16_Matrix @ funcValuesAtNodes / h
    funcDerivativeValuesAtNodes = funcDerivative(nodes)
    FD2_ERROR = np.max(np.abs(derivative2_ApproxSolution - funcDerivativeValuesAtNodes))
    FD4_ERROR = np.max(np.abs(derivative4_ApproxSolution - funcDerivativeValuesAtNodes))
    FD8_ERROR = np.max(np.abs(derivative8_ApproxSolution - funcDerivativeValuesAtNodes))
    FD16_ERROR = np.max(np.abs(derivative16_ApproxSolution - funcDerivativeValuesAtNodes))

    D = spec.chebDiffMatrix(n, -1, 1)
    # np.set_printoptions(precision=3, suppress=True)
    # print(D)
    # I = np.eye(n)
    x = spec.chebNodes(n, -1, 1)
    # print(x - np.pi)
    # sol = sp_lin.solve(D + I, funcDerivative(x - np.pi))
    # SPECTRAL_ERROR = np.max(np.abs((D + I) @ func(x - np.pi) - funcDerivative(x - np.pi)))
    SPECTRAL_ERROR = np.max(np.abs(D @ func(x) - funcDerivative(x)))
    print(SPECTRAL_ERROR)
    # plt.plot(x, sol)
    # plt.plot(x, func(x))
    # plt.show()
    # return SPECTRAL_ERROR
    maxDerivativeValue = np.max(np.abs(funcDerivative(x)))
    return np.array([FD2_ERROR, FD4_ERROR, FD8_ERROR, FD16_ERROR, SPECTRAL_ERROR], dtype=float)/maxDerivativeValue

# np.set_printoptions(precision=3, suppress=True)

# compareDerivativeCalculation(20,
#                              lambda x: (1 + x)*np.exp(np.sin(10*x)**3 + np.cos(6*x)),
#                              lambda x: np.exp(np.sin(10*x)**3 + np.cos(6*x))*(1 - 6*(1 + x) * np.sin(6*x) + 15 * (1 + x) * np.sin(10*x)* np.sin(20*x)))
errors_list = []
indices_list = []
for i in range(3, 100, 1):
    # errors = derivativeError_WithDifferentMethods(i,
    #                          lambda x: (1 + x)*np.exp(np.sin(10*x)**3 + np.cos(6*x)),
    #                          lambda x: np.exp(np.sin(10*x)**3 + np.cos(6*x))*(1 - 6*(1 + x) * np.sin(6*x) + 15 * (1 + x) * np.sin(10*x)* np.sin(20*x)))
    errors = firstOrderODE_Zero_Dirichlet(i,
                                                  lambda x: (1 - x) * np.sin(x)**3,
                                                  lambda x: -np.sin(x)**2 * (3 * (-1 + x) * np.cos(x) + np.sin(x)),
                                                  lambda x: 0)


    indices_list.append(errors[0])
    errors_list.append(errors[1])
fig, ax = plt.subplots()
line_styles = ['-', '--', '-.', ':', '-']
markers = ['o', 's', 'D', 'v', '^']

# indices_lists = np.array(indices_list)

# indices_lists = [
#     (5 - 1) * (2 * (1 + 5) * (indices_list - 1) + indices_list),
#     (9 - 1) * (2 * (1 + 9) * (indices_list - 1) + indices_list),
#     (17 - 1) * (2 * (1 + 17) * (indices_list - 1) + indices_list),
#     (33 - 1) * (2 * (1 + 33) * (indices_list - 1) + indices_list),
#     1.0/6.0 * indices_list * (13 + 4 * indices_list) * (indices_list - 1)
# ]
# for it in indices_lists:
#     print(it[0])

# Plot the data with distinct line styles and markers
for i, label in enumerate(['FD2_ERROR', 'FD4_ERROR', 'FD8_ERROR', 'FD16_ERROR', 'SPECTRAL_ERROR']):
    ax.loglog([index[i] for index in indices_list], [error[i] for error in errors_list], label=label, linestyle=line_styles[i])

# Customize the plot
ax.set_title('Error Analysis')
ax.set_xlabel('Indices')
ax.set_ylabel('Error')
ax.legend(loc='best')


# custom_x_ticks = [20, 40, 80, 160, 320]
# ax.set_xticks(custom_x_ticks)

# Format the x-axis labels in scientific notation
# x_labels = ['$10 \cdot 2^{' + str(i) + '}$' for i in range(1, 6)]
# ax.set_xticklabels(x_labels)
custom_y_ticks = [10**(-i) for i in range(10)]
ax.set_yticks(custom_y_ticks)

# Format the y-axis labels in scientific notation
y_labels = [f'$10^{{-{i}}}$' for i in range(10)]
ax.set_yticklabels(y_labels)


ax.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()



# compareDerivativeCalculation(40,
#                              lambda x: (1 + x)*np.exp(np.sin(10*x)**3 + np.cos(6*x)),
#                              lambda x: np.exp(np.sin(10*x)**3 + np.cos(6*x))*(1 - 6*(1 + x) * np.sin(6*x) + 15 * (1 + x) * np.sin(10*x)* np.sin(20*x)))
# compareDerivativeCalculation(80,
#                              lambda x: (1 + x)*np.exp(np.sin(10*x)**3 + np.cos(6*x)),
#                              lambda x: np.exp(np.sin(10*x)**3 + np.cos(6*x))*(1 - 6*(1 + x) * np.sin(6*x) + 15 * (1 + x) * np.sin(10*x)* np.sin(20*x)))
# compareDerivativeCalculation(160,
#                              lambda x: (1 + x)*np.exp(np.sin(10*x)**3 + np.cos(6*x)),
#                              lambda x: np.exp(np.sin(10*x)**3 + np.cos(6*x))*(1 - 6*(1 + x) * np.sin(6*x) + 15 * (1 + x) * np.sin(10*x)* np.sin(20*x)))