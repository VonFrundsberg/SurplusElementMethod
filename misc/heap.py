from FiniteElementMethod.element.mainElementClass import element
import FiniteElementMethod.element.elementOperations as operations
import scipy.sparse as sp_sparse
import numpy.linalg as np_lin
import scipy.sparse.linalg as sparse_linalg
from mathematics import approximate as approx
from mathematics import integrate as integr
from mathematics import spectral as spec
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sp_lin


def permutations(n):
    """Construct array of all permutations of arange(n) array"""
    a = np.zeros((np.math.factorial(n), n), np.uint8)
    f = 1
    for m in range(2, n + 1):
        b = a[:f, n - m + 1:]  # the block of permutations of range(m-1)
        for i in range(1, m):
            a[i * f:(i + 1) * f, n - m] = i
            a[i * f:(i + 1) * f, n - m + 1:] = b + (b >= i)
        b += 1
        f *= m
    return a
def testKronSum(elementU):
    """Some tests with kronecker sum of a differential-like operator, conversion to TT train format"""
    basisFuncList = []
    basisDiffFuncList = []
    for i in range(elementU.getDim()):
        basisFuncList.append(elementU[i].evalAtChebPoints()[1:-1, 1: -1])
        basisDiffFuncList.append(elementU[i].evalDiffRefNodes()[1:-1, 1: -1])
    C = np.kron(np.kron(basisDiffFuncList[0], basisFuncList[1]), basisFuncList[2])
    C += np.kron(np.kron(basisFuncList[0], basisDiffFuncList[1]), basisFuncList[2])
    C += np.kron(np.kron(basisFuncList[0], basisFuncList[1]), basisDiffFuncList[2])
    C += np.kron(np.kron(basisFuncList[0], basisFuncList[1]), basisFuncList[2])

    print(C.shape)
    old_C = C.copy()
    C = np.reshape(C, np.hstack((elementU.approxOrder - 2, elementU.approxOrder - 2)))
    C = np.transpose(C, axes=[0, 3, 1, 4, 2, 5])
    # C = np.transpose(C, axes=[0, 4, 1, 5, 2, 6, 3, 7])
    C = np.reshape(C, (elementU.approxOrder - 2)**2)
    C_TT_kron = approx.kronSumtoTT(basisFuncList, basisDiffFuncList)
    # for i in range(3):
    #     print(C_TT_1[i].shape)
    #     shape = C_TT_1[i].shape
    #     C_TT_1[i] = np.reshape(C_TT_1[i], [shape[0], shape[1]*shape[2], shape[3]])
    # if i == 0:
    #     print(i, "s core, upper")
    #     print(np.reshape(C_TT_2[i][0, :, 0], [3, 3]))
    #     print(i, "s core lower")
    #     print(np.reshape(C_TT_2[i][0, :, 1], [3, 3]))
    # shape = C_TT_2[i].shape
    # C_TT_2[i] = np.reshape(C_TT_2[i], [shape[0], shape[1] * shape[2], shape[3]])
    # print(C_TT_1[i].shape)
    # C = np.reshape(C, (elementU.approxOrder - 2)**2)
    # A = np.reshape(C_TT_1[0], [C_TT_1[0].shape[1], C_TT_1[0].shape[-1]])
    # initShape = C.shape
    # for i in range(1, len(C_TT_1)):
    #     shape = C_TT_1[i].shape
    #     reshaped = np.reshape(C_TT_1[i], [shape[0], shape[1] * shape[2]])
    #     A = np.dot(A, reshaped)
    #     A = np.reshape(A, [np.prod(initShape[:i + 1]), int(A.shape[-1]/initShape[i])])
    # A = np.reshape(A, initShape)
    # print("first", np.sum((A - C)**2))

    # A = np.reshape(C_TT_1[0], [C_TT_1[0].shape[1], C_TT_1[0].shape[-1]])
    # initShape = C.shape
    # for i in range(1, len(C_TT_1)):
    #     shape = C_TT_1[i].shape
    #     reshaped = np.reshape(C_TT_1[i], [shape[0], shape[1] * shape[2]])
    #     A = np.dot(A, reshaped)
    #     A = np.reshape(A, [np.prod(initShape[:i + 1]), int(A.shape[-1] / initShape[i])])
    # A = np.reshape(A, initShape)
    # print("second", np.sum((A - C) ** 2))
    # C = np.reshape(C, old_C.shape)
    # print("just for test", np.max(C - old_C))
    #
    # approx.kronSumtoTT()
def testTT_approximation(elementU, weight, TT_Tolerance):
    grid = elementU.getGrid()
    weightArr = weight(grid)
    weightArrTT = approx.simpleTTsvd(weightArr, tol=TT_Tolerance)
    print("elements amount before TT ", np.prod(weightArr.shape))
    sum = 0
    for it in weightArrTT:
        # print(it.shape, np.prod(it.shape))
        sum += np.prod(it.shape)
    A = np.reshape(weightArrTT[0], [weightArrTT[0].shape[1], weightArrTT[0].shape[-1]])
    initShape = weightArr.shape
    for i in range(1, len(weightArrTT)):
        shape = weightArrTT[i].shape
        reshaped = np.reshape(weightArrTT[i], [shape[0], shape[1] * shape[2]])
        A = np.dot(A, reshaped)
        A = np.reshape(A, [np.prod(initShape[:i + 1]), int(A.shape[-1]/initShape[i])])
    A = np.reshape(A, initShape)
    print(np.sum((A - weightArr)**2))
    print("after TT ", sum)

    binWeightArrTT = approx.simpleQTTsvd(weightArr, tol=1e-6)
    sum = 0
    for it in binWeightArrTT:
        # print(it.shape, np.prod(it.shape))
        sum += np.prod(it.shape)

    A = np.reshape(binWeightArrTT[0], [2, binWeightArrTT[0].shape[-1]])

    initShape = weightArr.shape
    for i in range(1, len(binWeightArrTT)):
        shape = binWeightArrTT[i].shape
        reshaped = np.reshape(binWeightArrTT[i], [shape[0], shape[1] * shape[2]])
        A = np.dot(A, reshaped)
        A = np.reshape(A, [2 ** (i + 1), int(A.shape[-1] / 2)])
    A = np.reshape(A, initShape)
    print(np.sum(np.abs(A - weightArr)))
    print("after binaryTT ", sum)
def testLaplacianInverseTT(elementU):
    basisFuncList = []
    basisDiffFuncList = []
    for i in range(elementU.getDim()):
        basisFuncList.append(elementU[i].evalAtChebPoints()[1:-1, 1: -1])
        basisDiffFuncList.append(elementU[i].evalDiffRefNodes()[1:-1, 1: -1])
    C = np.kron(np.kron(np.dot(basisDiffFuncList[0], basisDiffFuncList[0]), basisFuncList[1]), basisFuncList[2])
    C += np.kron(np.kron(basisFuncList[0], np.dot(basisDiffFuncList[1], basisDiffFuncList[1])), basisFuncList[2])
    C += np.kron(np.kron(basisFuncList[0], basisFuncList[1]), np.dot(basisDiffFuncList[2], basisDiffFuncList[2]))
    invC = sp_lin.inv(C)
    shape = elementU.approxOrder
    ttC = approx.matrixTTsvd(C, shape - 2, tol=1e-6)
    ttInvC = approx.matrixTTsvd(invC, shape - 2, tol=1e-3)
    print("nonzero amount in kron matrix: ", np.count_nonzero(C))
    print("nonzero amount in inverse matrix: ", np.count_nonzero(invC))
    sum = 0
    for i in range(len(ttC)):
        print("C core shape:", ttC[i].shape, "inverse C core shape ", ttInvC[i].shape, "with the total size"
                                                                                       "of ", ttInvC[i].size)
        sum += ttInvC[i].size
    print("inverse and TT approximation nonzero elements magnitude difference (times):", np.count_nonzero(invC)/sum)
    for i in range(3):
        shape = ttInvC[i].shape
        ttInvC[i] = np.reshape(ttInvC[i], [shape[0], shape[1]*shape[2], shape[3]])

        shape = ttC[i].shape
        ttC[i] = np.reshape(ttC[i], [shape[0], shape[1] * shape[2], shape[3]])

    newCshape = np.array(elementU.approxOrder - 2, dtype=int)
    invC = np.reshape(invC, [*newCshape, *newCshape])
    invC = np.transpose(invC, axes=[0, 3, 1, 4, 2, 5])
    invC = np.reshape(invC, newCshape**2)
    A = np.reshape(ttInvC[0], [ttInvC[0].shape[1], ttInvC[0].shape[-1]])
    initShape = invC.shape
    for i in range(1, len(ttInvC)):
        shape = ttInvC[i].shape
        reshaped = np.reshape(ttInvC[i], [shape[0], shape[1] * shape[2]])
        A = np.dot(A, reshaped)
        A = np.reshape(A, [np.prod(initShape[:i + 1]), int(A.shape[-1]/initShape[i])])
    A = np.reshape(A, initShape)
    print("approximation error for inverse: ", np.sum((A - invC)**2))

    newCshape = np.array(elementU.approxOrder - 2, dtype=int)
    C = np.reshape(C, [*newCshape, *newCshape])
    C = np.transpose(C, axes=[0, 3, 1, 4, 2, 5])
    C = np.reshape(C, newCshape ** 2)
    A = np.reshape(ttC[0], [ttC[0].shape[1], ttC[0].shape[-1]])
    initShape = C.shape
    for i in range(1, len(ttC)):
        shape = ttC[i].shape
        reshaped = np.reshape(ttC[i], [shape[0], shape[1] * shape[2]])
        A = np.dot(A, reshaped)
        A = np.reshape(A, [np.prod(initShape[:i + 1]), int(A.shape[-1] / initShape[i])])
    A = np.reshape(A, initShape)
    print("approximation error for original matrix: ", np.sum((A - C) ** 2))
def solvePeriodicODE(polyOrder, rhsF, halfInterval=False):
    """Numerically solves the equation
        du + u = rhsF(x), where rhsF is a periodic function,
        with halfInterval false the interval is [0, 2*pi], with true it is [0, pi]"""
    n = polyOrder
    D = spec.periodicDiffMatrix(n, halfInterval)
    I = np.eye(n)
    x = spec.periodicNodes(n, halfInterval)
    sol = sp_lin.solve(D + I, rhsF(x))
    return sol
def solvePrintPlotPeriodicODE(polyorder, rhsF, asol):
    sol = solvePeriodicODE(polyorder, rhsF, True)
    x = spec.periodicNodes(polyorder, halfInterval=True)
    print('approximation order is: ', polyorder)
    print(sol - asol(x))
    plt.plot(x, sol)
    plt.plot(x, asol(x))
    plt.show()
import time
def solveSphericalPois(polyOrder, rhsF, integrPoints=2000):

    elem = element(np.array([[0, np.inf], [0, np.pi], [0, 2*np.pi]]),
                   np.array(polyOrder, dtype=int), np.array([1, 0, 3]))

    rMatrixD = operations.integrateBilinearForm1(elem, lambda x: x * x, integrPoints, 0)[:-1, :-1]
    rMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 0)[:-1, :-1]

    # print(rMatrixI)

    # tMatrixD = elem[1].evalDiffRefNodes()
    tMatrixD = operations.integrateBilinearForm1(elem, lambda x: np.sin(x)**2, integrPoints, 1) + \
               operations.integrateBilinearForm2(elem, lambda x: np.sin(x)*np.cos(x), integrPoints, 1)
    # #
    tMatrixIr = operations.integrateBilinearForm0(elem, lambda x: np.sin(x)**2, integrPoints, 1)
    tMatrixIp = operations.integrateBilinearForm0(elem, lambda x: x*0 + 1.0, integrPoints, 1)

    # tMatrixD = operations.integrateBilinearForm1(elem, lambda x: np.sin(x), integrPoints, 1)
    # tMatrixIr = operations.integrateBilinearForm0(elem, lambda x: np.sin(x), integrPoints, 1)
    # tMatrixIp = operations.integrateBilinearForm0(elem, lambda x: 1.0 / np.sin(x), integrPoints, 1)
    # tMatrixIp[0, 0] = np.inf
    # print(tMatrixIp)
    # time.sleep(500)
    # t = elem[1].getMappedRefPoints()
    # tDiag1 = np.diag((np.sin(t))*np.cos(t))
    # tDiag2 = np.diag(np.sin(t)**2)
    # tMatrixD = tDiag2 @ spec.periodic2DiffMatrix(elem[1].approxOrder)
    # tMatrixD += tDiag1 @ spec.periodicDiffMatrix(elem[1].approxOrder)
    # print(np.dot(tMatrixD, np.sin(t)) - np.sin(t)*(np.cos(t)**2 - np.sin(t)**2))

    # tMatrixIr = tDiag2 @ np.eye(elem[1].approxOrder)
    # tMatrixIp = np.eye(elem[1].approxOrder)

    # pMatrixD = elem[2].evalDiffRefNodes()
    # pMatrixD = np.dot(pMatrixD, pMatrixD)
    # p = elem[2].getMappedRefPoints()
    pMatrixD = spec.periodic2DiffMatrix(elem[2].approxOrder)
    pMatrixI = np.eye(elem[2].approxOrder)

    pMatrixD = spec.periodic2DiffMatrix(elem[2].approxOrder)
    pMatrixI = np.eye(elem[2].approxOrder)
    # print(np.dot(pMatrixD, np.sin(2*p)) + 4*np.sin(2*p))
    # time.sleep(500)
    # print(np_lin.cond(rMatrixD), np_lin.cond(tMatrixD), np_lin.cond(pMatrixD))

    # basisFuncList = [rMatrixI, None, pMatrixI]
    # basisDiffFuncList = [rMatrixD, tMatrixD, pMatrixD]

    C = -sp_sparse.kron(sp_sparse.kron(rMatrixD, tMatrixIr), pMatrixI)
    C -= sp_sparse.kron(sp_sparse.kron(rMatrixI, tMatrixD), pMatrixI)
    C += sp_sparse.kron(sp_sparse.kron(rMatrixI, tMatrixIp), pMatrixD)
    # C.data[np.isinf(C.data)] = 0.0
    # C = sp_sparse.kron(sp_sparse.kron(pMatrixD, tMatrixIp), rMatrixI)
    # C += sp_sparse.kron(sp_sparse.kron(pMatrixI, tMatrixD), rMatrixI)
    # C += sp_sparse.kron(sp_sparse.kron(pMatrixI, tMatrixIr), rMatrixD)

    grid = elem.getGridList()
    w, idNodes = integr.reg_32_wn(-1, 1, 500)
    grid[0] = elem[0].map(idNodes)
    grid[1] = elem[1].map(idNodes)
    grid = approx.meshgrid(*grid)

    fx = rhsF(grid)
    fx = np.nan_to_num(fx, 0)

    ttFx = approx.simpleTTsvd(fx, tol=1e-6)
    # for i in ttFx:
    #     print(i.shape)
    ttR = ttFx[0].copy()
    preshape = ttR.shape
    ttR = np.reshape(ttR, [ttR.shape[1], ttR.shape[2]])
    integral = operations.integrateFunctional(elem, ttR, integrPoints, 0, True, precalc=True)
    integral = integral[np.newaxis, :-1, :]
    ttFx[0] = integral

    ttT = ttFx[1].copy()
    # print(ttT.shape)
    preshape = ttT.shape
    # ttT = np.reshape(ttT, [ttT.shape[0], ttT.shape[1]])
    ttT = np.transpose(ttT, [1, 0, 2])
    ttT = np.reshape(ttT, [preshape[1], preshape[0]*preshape[2]])
    integral = operations.integrateFunctional(elem, ttT, integrPoints, 1, True, precalc=True)
    integral = integral[:, :, np.newaxis]
    integral = np.reshape(integral, [elem[1].approxOrder, preshape[0], preshape[2]])
    integral = np.transpose(integral, [1, 0, 2])
    # print(integral.shape)
    ttFx[1] = integral
    # print(integral.shape)
    print(approx.toFullTensor(ttFx).shape)
    fx = (approx.toFullTensor(ttFx)).flatten()
    # print(np_lin.cond(C.toarray()))
    # print("everything is calculated, starting system solving")
    sol = sparse_linalg.spsolve(C, fx)
    print("max is: ", np.max(np.abs(sol)))
    sol = np.reshape(sol, elem.approxOrder - [1, 0, 0])
    ttFx = approx.simpleTTsvd(sol, tol=1e-6)
    for i in ttFx:
        print(i.shape)

    # print(np.max(np.abs(fx - sol)))


    # subplot(r,c) provide the no. of rows and columns
    r, t, p = elem.getGridList()
    # print(np.cos(2*t))
    rr, tt, pp = approx.meshgrid(r[:-1], t, p)
    fx = -np.exp(-rr) * np.cos(2 * tt) * np.sin(pp)
    # print("shapppe", fx.shape)
    reshaped = np.reshape(fx.flatten(), elem.approxOrder - [1, 0, 0])
    # print("max is", np.max(np.abs(reshaped - fx)))
    # print(fx.shape)
    # plt.figure()
    f, axarr = plt.subplots(elem[1].approxOrder, 2)
    for i in range(elem[1].approxOrder):
        axarr[i, 0].imshow(sol[:, i, :] + fx[:, i, :])
        # axarr[i, 1].imshow(fx[:, i, :])
        # plt.show()
    #     plt.plot(sol[:, :, i].T)
    #     plt.colorbar()
    # plt.colorbar()
    plt.show()
        # plt.imshow()
        # plt.show()
    # if integral.shape[1] == 1:
    #     integral = np.squeeze(integral)

    # ttFx[0] =




    # print(C.toarray().size)
    # ttC = approx.matrixTTsvd(C.toarray(), polyOrder - [1, 0, 0])
    # for it in ttC:
    #     print(it.shape)

def solveSphericalPoisNoPhi(polyOrder, rhsF, integrPoints=2000):

    elem = element(np.array([[0, np.inf], [0, np.pi]]),
                   np.array(polyOrder, dtype=int), np.array([1, 0]))

    rMatrixD = operations.integrateBilinearForm1(elem, lambda x: x * x, integrPoints, 0)[:-1, :-1]
    rMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 0)[:-1, :-1]

    # tMatrixD = operations.integrateBilinearForm1(elem, lambda x: np.sin(x), integrPoints, 1)
    tMatrixD = operations.integrateBilinearForm1(elem, lambda x: np.sin(x) ** 2, integrPoints, 1) + \
               operations.integrateBilinearForm2(elem, lambda x: np.sin(x) * np.cos(x), integrPoints, 1)
    tMatrixI = operations.integrateBilinearForm0(elem, lambda x: np.sin(x)**2, integrPoints, 1)

    # print(np_lin.cond(rMatrixD), np_lin.cond(tMatrixD))


    C = -sp_sparse.kron(rMatrixD, tMatrixI)
    C -= sp_sparse.kron(rMatrixI, tMatrixD)

    grid = elem.getGridList()
    w, idNodes = integr.reg_32_wn(-1, 1, 500)
    grid[0] = elem[0].map(idNodes)
    grid[1] = elem[1].map(idNodes)
    grid = approx.meshgrid(*grid)

    fx = rhsF(grid)
    fx = np.nan_to_num(fx, 0)

    ttFx = approx.simpleTTsvd(fx, tol=1e-6, R_MAX=1000)
    # for i in ttFx:
    #     print(i.shape)



    ttR = ttFx[0].copy()
    preshape = ttR.shape
    ttR = np.reshape(ttR, [ttR.shape[1], ttR.shape[2]])
    integral = operations.integrateFunctional(elem, ttR, integrPoints, 0, True, precalc=True)
    integral = integral[np.newaxis, :-1, :]
    # integral = np.reshape(integral[np.newaxis, :-1, :], np.array([1, elem[0].approxOrder, preshape[2]]) - np.array([0, 1, 0]))
    ttFx[0] = integral

    ttT = ttFx[1].copy()
    preshape = ttT.shape
    ttT = np.reshape(ttT, [ttT.shape[0], ttT.shape[1]]).T
    integral = operations.integrateFunctional(elem, ttT, integrPoints, 1, True, precalc=True)
    integral = integral[:, :, np.newaxis]
    integral = np.transpose(integral, [1, 0, 2])
    # integral = np.reshape(integral[:, :, np.newaxis],
    #                       np.array([preshape[0], elem[1].approxOrder, preshape[2]]))
    ttFx[1] = integral

    fx = approx.toFullTensor(ttFx).flatten()
    print("everything is calculated, starting system solving")
    sol = sparse_linalg.spsolve(C, fx)
    print("max is: ", np.max(np.abs(sol)))
    sol = np.reshape(sol, elem.approxOrder - [1, 0])
    ttFx = approx.simpleTTsvd(sol, tol=1e-6)
    # for i in ttFx:
    #     print(i.shape)
    # f, axarr = plt.subplots(2, 1)
    r, t = elem.getGridList()
    # print(np.cos(2*t))
    rr, tt = approx.meshgrid(r[:-1], t)
    fx = -np.exp(-rr) * np.cos(2 * tt)
    plt.imshow(sol + fx)
    # axarr[1].imshow(fx)
            # plt.show()
        #     plt.plot(sol[:, :, i].T)
        #     plt.colorbar()
    plt.colorbar()
    plt.show()
    # r, t, p = elem.getGrid()
    # fx = 1.0/(1 + r)*np.sin(t)*np.cos(2*p)
    # for i in range(elem[2].approxOrder):
    #     print(np.max(np.abs(sol[:, :, i] + fx[:-1, :, i])))
    #     plt.imshow(sol[:, :, i])
    #     plt.plot(sol[:, :, i].T)
        # plt.colorbar()
        # plt.show()
    # if integral.shape[1] == 1:
    #     integral = np.squeeze(integral)

    # ttFx[0] =




    # print(C.toarray().size)
    # ttC = approx.matrixTTsvd(C.toarray(), polyOrder - [1, 0, 0])
    # for it in ttC:
    #     print(it.shape)
# def f(x):
#     return np.exp(-np.sqrt(4*x[0]**2 + 2*x[1]**2 + x[2]**2))
np.set_printoptions(precision=5, suppress=True)
# for i in range(4, 20, 2):
#     solveSphericalPoisNoPhi(np.array([2*i, i]),
#                        # lambda x: (np.sin(x[1])**2)*np.exp(-x[0])*np.sin(x[2])*(np.cos(x[1])*\
#                        #              (6.0 - (-2.0 + x[0])*x[0]) + 1.0/(np.sin(x[1])**2)))
#                        # lambda x: (np.sin(x[1])**2)*np.cos(x[2])/((1+x[0])**3)*(-2*(3+x[0]*(7+3*x[0]))*np.cos(2*x[1]) - \
#                        #                                    (1 + x[0])**2/(np.sin(x[1])**2)))
#                        #  lambda x: (x[0]**2)*(np.sin(x[1])*np.sin(x[1]))*np.cos(2*x[1])*np.exp(-x[0]))#*\
#                         lambda x: (np.sin(x[1])**2)*np.exp(-x[0])*(-2.0 + (-6.0 + (-2.0 + x[0])*x[0])*np.cos(2.0*x[1])))
for i in range(6, 14, 2):
    solveSphericalPois(np.array([2*i, i + 1, i]),
                       # lambda x: (np.sin(x[1])**2)*np.exp(-x[0])*np.sin(x[2])* \
                       #           ((6.0 - (-2.0 + x[0])*x[0])*np.cos(2*x[2]) + 1.0/(np.sin(x[1])**2)))
    # time.sleep(500)
    #                    lambda x: (np.sin(x[1])**2)*np.cos(x[2])/((1+x[0])**3)*(-2*(3+x[0]*(7+3*x[0]))*np.cos(2*x[1]) - \
    #                    #                                    (1 + x[0])**2/(np.sin(x[1])**2)))
    #                     lambda x: (x[0]**2)*(np.sin(x[1])*np.sin(x[1]))*np.cos(2*x[1])*np.exp(-x[0]))#*\
    #                              np.sin(x[1])*np.cos(x[2]))
                       #THIS ONE IS GOOD
                       lambda x: np.sin(x[1])*np.sin(x[1]) * np.exp(-x[0]) * (
                                   -2.0 + (-6.0 + (-2.0 + x[0]) * x[0]) * np.cos(2.0 * x[1])))

# for i in range(10, 14):
#     print(i, " i")
#     solveSphericalPois(np.array([i*2, i, i]),
#                        lambda x: (np.sin(x[1]))*
#             (-4 - x[0]*(9 + 4*x[0]) + ((1 + x[0]*(3 + x[0]))*np.cos(2*x[1])))*np.cos(2*x[2])/((1+x[0])**3))
    # solveSphericalPois(np.array([2*i, i, i]),
    #                    lambda x: (np.sin(x[1])**2) * np.sin(x[1]*2) * np.cos(x[2]) *\
    #                              (2*(3 + x[0]*(7 + 3*x[0]))) / ((1 + x[0]) ** 3))
# testKronSum(elem)
# testTT_approximation(elem, f, 1e-6)
# testLaplacianInverseTT(elem)

