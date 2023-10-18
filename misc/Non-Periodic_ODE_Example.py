import numpy as np
import mathematics.spectral as spec
import scipy.linalg as sp_lin
import matplotlib.pyplot as plt
import time as time
def compareDerivativeCalculation(nodesAmount: int, func, funcDerivative):
    n = nodesAmount
    zerosMatrix = np.zeros([n, n], dtype=float)
    FD2_Matrix = zerosMatrix + 0.5*np.diag(np.ones(n - 1), k=1)
    FD2_Matrix -= np.transpose(FD2_Matrix)
    FD2_Matrix[0, :3] = [-3.0/2.0, + 2.0, - 1.0/2.0]
    FD2_Matrix[-1, -3:] = -FD2_Matrix[0, :3][::-1]


    FD4_Matrix = (zerosMatrix +
                  2.0/3.0 * np.diag(np.ones(n - 1), k=1) -
                  1.0/12.0 * np.diag(np.ones(n - 2), k=2))
    FD4_Matrix -= np.transpose(FD4_Matrix)

    FD4_Matrix[0, :5] = [-25/12, 4, -3, 4/3, -1/4]
    FD4_Matrix[1, :5] = [-1 / 4, -5/6, 3 / 2, -1/2, 1/12]

    for i in range(2):
        FD4_Matrix[-1 - i, -5:] = -FD4_Matrix[i, :5][::-1]


    FD8_Matrix = (zerosMatrix +
                  4.0 / 5.0 * np.diag(np.ones(n - 1), k=1) +
                  -1.0 / 5.0 * np.diag(np.ones(n - 2), k=2) +
                  4.0 / 105.0 * np.diag(np.ones(n - 3), k=3) +
                  -1.0 / 280.0 * np.diag(np.ones(n - 4), k=4))
    FD8_Matrix -= np.transpose(FD8_Matrix)

    FD8_Matrix[0, :9] = [-761 / 280, 8, -14, 56/3, -35/2, 56/5, -14/3, 8/7, -1/8]
    FD8_Matrix[1, :9] = [-1 / 8, -223/140, 7 / 2, -7 / 2, 35 / 12, -7/4, 7/10, -1/6, 1/56]
    FD8_Matrix[2, :9] = [1/56, -2/7, -19/20, 2, -5/4, 2/3, -1/4, 2/35, -1/168]
    FD8_Matrix[3, :9] = [-1 / 168, 1/14, -1/2, -9 / 20, 5/4, -1/2, 1/6, -1/28, 1/280]

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
    FD16_Matrix[0, :17] = [-3.3807289932289932, 16, -60, 186.66666666666666, -455, 873.6, -1334.6666666666667, 1634.2857142857142, -1608.75,
     1271.111111111111, -800.8, 397.09090909090907, -151.66666666666666, 43.07692307692308, -8.571428571428571,
     1.0666666666666667, -0.0625]
    FD16_Matrix[1, :17] = [-0.0625, -2.3182289932289932, 7.5, -17.5, 37.916666666666664, -68.25, 100.1, -119.16666666666667,
     114.91071428571429, -89.375, 55.611111111111114, -27.3, 10.340909090909092, -2.9166666666666665,
     0.5769230769230769, -0.07142857142857142, 0.004166666666666667]
    FD16_Matrix[2, :17] = [0.004166666666666667, -0.13333333333333333, -1.7515623265623266, 4.666666666666667, -7.583333333333333,
     12.133333333333333, -16.683333333333334, 19.066666666666666, -17.875, 13.619047619047619, -8.341666666666667,
     4.044444444444444, -1.5166666666666666, 0.42424242424242425, -0.08333333333333333, 0.010256410256410256,
     -0.0005952380952380953]
    FD16_Matrix[3, :17] = [-0.0005952380952380953, 0.014285714285714285, -0.21428571428571427, -1.3468004218004217, 3.25, -3.9,
     4.766666666666667, -5.107142857142857, 4.5964285714285715, -3.4047619047619047, 2.0428571428571427, -0.975,
     0.3611111111111111, -0.1, 0.01948051948051948, -0.002380952380952381, 0.00013736263736263736]
    FD16_Matrix[4, :17] = [0.00013736263736263736, -0.0029304029304029304, 0.03296703296703297, -0.3076923076923077, -1.0198773448773448, 2.4,
     -2.2, 2.0952380952380953, -1.7678571428571428, 1.2571428571428571, -0.7333333333333333, 0.34285714285714286,
     -0.125, 0.03418803418803419, -0.006593406593406593, 0.0007992007992007992, -0.00004578754578754579]
    FD16_Matrix[5, :17] = [-0.00004578754578754579, 0.0009157509157509158, -0.009157509157509158, 0.0641025641025641, -0.4166666666666667,
     -0.7365440115440115, 1.8333333333333333, -1.3095238095238095, 0.9821428571428571, -0.6547619047619048,
     0.36666666666666664, -0.16666666666666666, 0.05952380952380952, -0.016025641025641024, 0.0030525030525030525,
     -0.0003663003663003663, 0.000020812520812520813]
    FD16_Matrix[6, :17] = [0.000020812520812520813, -0.0003996003996003996, 0.0037462537462537465, -0.023310023310023312, 0.11363636363636363,
     -0.5454545454545454, -0.478968253968254, 1.4285714285714286, -0.8035714285714286, 0.47619047619047616, -0.25,
     0.10909090909090909, -0.03787878787878788, 0.00999000999000999, -0.0018731268731268732, 0.000222000222000222,
     -0.000012487512487512488]
    FD16_Matrix[7, :17] = [-0.000012487512487512488, 0.0002331002331002331, -0.002097902097902098, 0.012237762237762238, -0.05303030303030303,
     0.19090909090909092, -0.7, -0.2361111111111111, 1.125, -0.5, 0.23333333333333334, -0.09545454545454546,
     0.031818181818181815, -0.008158508158508158, 0.0014985014985014985, -0.00017482517482517483, 9.712509712509713e-6]
    for i in range(8):
        FD16_Matrix[-1 - i, -17:] = -FD16_Matrix[i, :17][::-1]




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


np.set_printoptions(precision=3, suppress=True)

compareDerivativeCalculation(20,
                             lambda x: (1 + x)*np.exp(np.sin(10*x)**3 + np.cos(6*x)),
                             lambda x: np.exp(np.sin(10*x)**3 + np.cos(6*x))*(1 - 6*(1 + x) * np.sin(6*x) + 15 * (1 + x) * np.sin(10*x)* np.sin(20*x)))
compareDerivativeCalculation(40,
                             lambda x: (1 + x)*np.exp(np.sin(10*x)**3 + np.cos(6*x)),
                             lambda x: np.exp(np.sin(10*x)**3 + np.cos(6*x))*(1 - 6*(1 + x) * np.sin(6*x) + 15 * (1 + x) * np.sin(10*x)* np.sin(20*x)))
compareDerivativeCalculation(80,
                             lambda x: (1 + x)*np.exp(np.sin(10*x)**3 + np.cos(6*x)),
                             lambda x: np.exp(np.sin(10*x)**3 + np.cos(6*x))*(1 - 6*(1 + x) * np.sin(6*x) + 15 * (1 + x) * np.sin(10*x)* np.sin(20*x)))
compareDerivativeCalculation(160,
                             lambda x: (1 + x)*np.exp(np.sin(10*x)**3 + np.cos(6*x)),
                             lambda x: np.exp(np.sin(10*x)**3 + np.cos(6*x))*(1 - 6*(1 + x) * np.sin(6*x) + 15 * (1 + x) * np.sin(10*x)* np.sin(20*x)))