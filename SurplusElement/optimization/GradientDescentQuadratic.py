import numpy as np
from typing import Callable
import numpy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import SurplusElement.mathematics.approximate as approx
import time as time
import scipy.linalg as sp_lin
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
    # x0 = np.ones(dim, dtype=float)
    xNext = x0.copy()
    if output == False:
        for i in range(maxIter):
            xPrev = xNext
            xNext = xPrev - alpha * (gammaGradFunc(xPrev))
    else:
        for i in range(maxIter):
            xPrev = xNext
            xNext = xPrev - alpha * (gammaGradFunc(xPrev))
            print("min: ", (xNext @ (A @ xNext))/(xNext @ (B @ xNext)))
    return xNext

def YunhoGradientDescentWithMomentum(A: np.array, B: np.array, x0: np.array, alpha: float, beta:float, gamma:float,
                                   eps: float=1e-6, output=True, maxIter: int = 100):
    constMatrix = (A + gamma * B)
    gammaGradFunc = lambda x: constMatrix @ x - (gamma * B @ x)/np.sqrt(x @ (B @ x))
    dim = A.shape[0]
    # x0 = np.ones(dim, dtype=float)
    xNext = x0.copy()
    xPrev = x0.copy()
    if output == False:
        for i in range(maxIter):
            xPrev2 = xPrev
            xPrev = xNext
            xNext = xPrev - alpha * (gammaGradFunc(xPrev)) + beta * (xPrev - xPrev2)
    else:
        for i in range(maxIter):
            xPrev2 = xPrev
            xPrev = xNext
            xNext = xPrev - alpha * (gammaGradFunc(xPrev)) + beta * (xPrev - xPrev2)
            # print((xNext @ (B @ xNext)))
            print("min: ", (xNext @ (A @ xNext))/(xNext @ (B @ xNext)))
    return xNext



def YunhoScipy(A: np.array, B: np.array, x0: np.array, gamma:float, method:str = 'CG'):
    constMatrix = (A/2.0 + gamma * B / 2.0)
    func = lambda x: x @ constMatrix @ x - gamma * np.sqrt(x @ (B @ x))
    gammaConstMatrix = (A + gamma * B)
    gammaGradFunc = lambda x: gammaConstMatrix @ x - (gamma * B @ x) / np.sqrt(x @ (B @ x))
    # func = lambda x: (x @ A @ x) / (x @ B @ x)
    result = minimize(fun=func, x0=x0, jac=gammaGradFunc, method="CG", options={'gtol': 1e-16, 'maxiter':1000000})
    # print(result)
    return result.x

def YunhoTensorTrain(Y:np.array, A: np.array, B: np.array, x0: np.array,
                     alpha:float, gamma:float,
                     maxRank: int, solTol: float = 1e-6, operatorMaxRank:int = 100, operatorTol:float = 1e-6,
                     eps: float = 1e-15, maxIter: int = 100, output: bool = False):
    # constMatrix = (A/2.0 + gamma * B / 2.0)
    sqrtDim = int(np.sqrt(A.shape[0]))
    # ttMatrix = approx.matrixTTsvd(constMatrix, np.array([sqrtDim, sqrtDim]), tol=1e-6)
    solution = np.reshape(x0, [sqrtDim, sqrtDim])
    ttVector = approx.vectorTTsvd(solution, tol=1e-6)
    ttY = approx.matrixTTsvd(Y, np.array([sqrtDim, sqrtDim]), tol=operatorTol, maxRank=operatorMaxRank, output=False)
    fullY = approx.toFullTensor(ttY, matrixForm=True)
    shapeFullY = fullY.shape
    fullY = np.reshape(fullY, [shapeFullY[0] * shapeFullY[1], shapeFullY[2] * shapeFullY[3]])
    ttA = approx.matrixTTsvd(A + fullY, np.array([sqrtDim, sqrtDim]), tol=operatorTol, maxRank=operatorMaxRank, output=False)
    # print(approx.printTT(ttA))
    # ttA = approx.sumTT2d(ttA, ttY)
    ttB = approx.matrixTTsvd(B, np.array([sqrtDim, sqrtDim]), tol=operatorTol, maxRank=operatorMaxRank, output=False)
    # print(approx.printTT(ttY))
    # print("matrix")
    # approx.printTT(ttMatrix)
    # print("vector")
    # approx.printTT(ttVector)
    # product = approx.matrixVectorProd(ttMatrix, ttVector, tol=1e-6)
    # print("product")
    # approx.printTT(product)
    # productFullForm = approx.toFullTensor(product).flatten()
    # print(np.dot(productFullForm, productFullForm))
    # print(approx.dotProd2d(product, product))
    # print(np.linalg.norm(productFullForm - constMatrix @ x0))

    # func = lambda x: x @ constMatrix @ x - gamma * np.sqrt(x @ (B @ x))
    # gammaConstMatrix = (A + gamma * B)
    # gammaGradFunc = lambda x: gammaConstMatrix @ x - (gamma * B @ x) / np.sqrt(x @ (B @ x))
    # func = lambda x: x @ constMatrix @ x - gamma * np.sqrt(x @ (B @ x))
    # gammaConstMatrix = (A + gamma * B)


    def gammaGradFunc(x):
        Ax = approx.matrixVectorProd(ttA, x, tol=solTol)
        Bx = approx.matrixVectorProd(ttB, x, tol=solTol)
        Bx[0] = Bx[0] * gamma
        Ax = approx.sumTT2d(Ax, Bx, tol=solTol)
        xBx = np.sqrt(approx.dotProd2dTT(x, Bx))
        Bx[0] = -1.0/xBx * Bx[0]
        return approx.sumTT2d(Ax, Bx, tol=solTol)

    # func = lambda x: (x @ A @ x) / (x @ B @ x)
    xNext = ttVector
    Ax = approx.matrixVectorProd(ttA, xNext)
    Bx = approx.matrixVectorProd(ttB, xNext)
    nextValue = approx.dotProd2dTT(xNext, Ax) / approx.dotProd2dTT(xNext, Bx)
    for i in range(maxIter):
        xPrev = xNext
        prevValue = nextValue
        prevGrad = gammaGradFunc(xPrev)
        prevGrad[0] = prevGrad[0] * (-alpha)
        # approx.vecRound(prevGrad, tol=1e-3)
        xNext = approx.sumTT2d(xPrev, prevGrad, tol=solTol)
        fullXNext = approx.toFullTensor(xNext)
        fullXNext = (fullXNext + fullXNext.T)/2.0
        xNext = approx.simpleTTsvd(fullXNext, maxRank=maxRank, output=False, tol=solTol)
        # approx.vecRound(xNext, R_MAX=1, output=False)
        # u, s, v = sp_lin.svd(fullXNext, full_matrices=False)
        # print(s)
        # approx.vecRound(xNext, tol=1)
        # approx.printTT(xNext)
        Ax = approx.matrixVectorProd(ttA, xNext, tol=solTol)
        Bx = approx.matrixVectorProd(ttB, xNext, tol=solTol)
        nextValue = approx.dotProd2dTT(xNext, Ax) / approx.dotProd2dTT(xNext, Bx)
        if np.abs(nextValue - prevValue) < eps:
            return xNext, approx.dotProd2dTT(xNext, Ax) / approx.dotProd2dTT(xNext, Bx)
        if output == True:
            print("min: ", approx.dotProd2dTT(xNext, Ax) / approx.dotProd2dTT(xNext, Bx))
    print("GD not converged")
    return xNext, approx.dotProd2dTT(xNext, Ax) / approx.dotProd2dTT(xNext, Bx)
    # result = minimize(fun=func, x0=x0, jac=gammaGradFunc, method="CG", options={'gtol': 1e-16, 'maxiter':1000000})
    # time.sleep(500)
    # print(result)
    # return result.x
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