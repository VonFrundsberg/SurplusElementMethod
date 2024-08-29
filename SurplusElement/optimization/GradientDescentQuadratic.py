import numpy as np
from typing import Callable
import numpy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
def minimizeOneQuadraticConstraint(A: np.array, B: np.array, x0: np.array, alpha: float,
                                   eps: float=1e-6, output=False, maxIter: int = 100):
    lambdaGradFunc = lambda l: (A.T + A) + l * (B.T + B)
    dim = A.shape[0]
    x0 = np.ones(dim, dtype=float)
    lNext: float = 1.0
    xNext = x0.copy()

    if output == False:
        for i in range(maxIter):
            xPrev = xNext
            lPrev = lNext
            xNext = xPrev - alpha * (lambdaGradFunc(lPrev) @ xPrev)
            lNext = lPrev - alpha * (xPrev @ (B @ xPrev) - 1.0)
    else:
        for i in range(maxIter):
            xPrev = xNext
            lPrev = lNext
            xNext = xPrev - alpha * (lambdaGradFunc(lPrev) @ xPrev)
            lNext = lPrev - alpha * (xPrev @ (B @ xPrev) - 1.0)

            print("min: ", xNext @ (A @ xNext), "constraint: ", xNext @ (B @ xNext))
            print("grad: ", np.linalg.norm(lambdaGradFunc(lNext) @ xNext))
            print("lambdaGrad: ", lNext, (xPrev @ (B @ xPrev) - 1.0))
    return xNext

def YunhoGradientDescent(A: np.array, B: np.array, x0: np.array, alpha: float, gamma:float,
                                   eps: float=1e-6, output=True, maxIter: int = 100):
    constMatrix = (A + gamma * B)
    gammaGradFunc = lambda x: constMatrix @ x - (gamma * B @ x)/np.sqrt(x @ (B @ x))
    dim = A.shape[0]
    x0 = np.ones(dim, dtype=float)
    xNext = x0.copy()
    if output == False:
        for i in range(maxIter):
            xPrev = xNext
    else:
        for i in range(maxIter):
            xPrev = xNext
            xNext = xPrev - alpha * (gammaGradFunc(xPrev))
            print("min: ", (xNext @ (A @ xNext))/(xNext @ (B @ xNext)))
    return
def minimizeOneQuadraticConstraintProjection(A: np.array, B: np.array, x0: np.array, alpha: float,
                                   eps: float=1e-6, output=True, maxIter: int = 100):
    lambdaGradFunc = (A.T + A)
    dim = A.shape[0]
    x0 = np.ones(dim, dtype=float)
    xNext = x0.copy()

    if output == False:
        for i in range(maxIter):
            xPrev = xNext
    else:
        for i in range(maxIter):
            xPrev = xNext
            xNext = xPrev - alpha * (lambdaGradFunc @ xPrev)
            constraintValue = xNext @ (B @ xNext)
            # plt.scatter(xPrev[0], xPrev[1], c='red')
            # plt.scatter(xNext[0], xNext[1])
            print("min: ", xNext @ (A @ xNext), "constraint: ", xNext @ (B @ xNext))
            xNext = xNext / np.sqrt(constraintValue)
            # plt.scatter(xNext[0], xNext[1], c='green')
            print("min: ", xNext @ (A @ xNext), "constraint: ", xNext @ (B @ xNext))
            # plt.plot(xNext)
            # plt.show()
            print(xNext)
            # print("grad: ", np.linalg.norm(lambdaGradFunc(lNext) @ xNext))
            # print("lambdaGrad: ", lNext, (xPrev @ (B @ xPrev) - 1.0))
    return xNext