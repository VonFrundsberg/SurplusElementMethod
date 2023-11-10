from FiniteElementMethod.element.mainElementClass import element
import FiniteElementMethod.element.elementUtils as operations
import scipy.sparse as sp_sparse
import numpy.linalg as np_lin
import scipy.sparse.linalg as sparse_linalg
from mathematics import approximate as approx
from mathematics import integrate as integr
from mathematics import spectral as spec
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sp_lin
from scikit_tt.tensor_train import TT

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

import time
def solveSphericalPois(polyOrder, rhsF, integrPoints=350):

    elem = element(np.array([[0, np.inf], [0, np.pi], [0, 2*np.pi]]),
                   np.array(polyOrder, dtype=int), np.array([1, 0, 3]))

    rMatrixD = operations.integrateBilinearForm1_SameElement(elem, lambda x: x * x, integrPoints, 0)[:-1, :-1]
    rMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 0)[:-1, :-1]

    tMatrixD = operations.integrateBilinearForm1_SameElement(elem, lambda x: np.sin(x) ** 2, integrPoints, 1) + \
               operations.integrateBilinearForm2(elem, lambda x: np.sin(x) * np.cos(x), integrPoints, 1)

    tMatrixIr = operations.integrateBilinearForm0(elem, lambda x: np.sin(x)**2, integrPoints, 1)
    tMatrixIp = operations.integrateBilinearForm0(elem, lambda x: x*0 + 1.0, integrPoints, 1)

    pMatrixD = operations.integrateBilinearForm1_SameElement(elem, lambda x: x * 0 + 1.0, integrPoints, 2)
    pMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 2)
    # print("here1?")
    # C = -np.kron(np.kron(rMatrixD, tMatrixIr), pMatrixI)
    # C -= np.kron(np.kron(rMatrixI, tMatrixD), pMatrixI)
    # C -= np.kron(np.kron(rMatrixI, tMatrixIp), pMatrixD)
    # print("here2?")
    # ttC = approx.matrixTTsvd(C.toarray(), elem.approxOrder - [1, 0, 0])
    # k = 1
    # i = 0
    # j = 0
    # print(ttC[i][j, :, :, k])
    ttC = approx.kronSumtoTT_blockFormat([[None, -rMatrixD, rMatrixI],
                                         [tMatrixIr, -tMatrixD, tMatrixIp],
                                         [pMatrixI, -pMatrixD, None]])
    # print(ttC[i][j, :, :, k])
    # time.sleep(500)
    # for it in ttC:
    #     print(it.shape)
    # ttC = approx.toFullTensor(ttC, matrixForm=True)
    # ttC = np.reshape(ttC, C.shape)
    # print(np.max(np.abs(ttC - C)))

    grid = elem.getGridList()
    w, idNodes = integr.reg_32_wn(-1, 1, integrPoints)
    grid[0] = elem[0].map(idNodes)
    # grid[1] = elem[1].map(idNodes)
    # grid[2] = elem[2].map(idNodes)
    grid = approx.meshgrid(*grid)

    fx = rhsF(grid)
    fx = np.nan_to_num(fx, 0)

    ttFx = approx.simpleTTsvd(fx, tol=1e-6, R_MAX=400)
    for it in ttFx:
        print(it.shape)
    time.sleep(500)
    # grid0 = idNodes
    # core0 = ttFx[0]
    # core0 = np.transpose(core0, [1, 0, 2])
    # core0 = spec.barycentricChebInterpolate(core0, grid0, a=-1, b=1, extrapolation=0, axis=0)
    # core0 = np.transpose(core0, [1, 0, 2])
    # ttFx[0] = core0

    grid1 = idNodes
    core1 = ttFx[1]
    core1 = np.transpose(core1, [1, 0, 2])
    core1 = spec.barycentricChebInterpolate(core1, grid1, a=-1, b=1, extrapolation=0, axis=0)
    core1 = np.transpose(core1, [1, 0, 2])
    ttFx[1] = core1
    # print(core1.shape)
    grid2 = idNodes
    core2 = ttFx[2]
    core2= np.transpose(core2, [1, 0, 2])
    core2 = np.einsum('ij, jnk -> ink', elem[2].eval(grid2), core2)

    core2 = np.transpose(core2, [1, 0, 2])
    ttFx[2] = core2

    # print('cores of f tt decomposition have the following shapes: ')
    # for i in ttFx:
    #     print(i.shape)
    # time.sleep(500)
    ttR = ttFx[0].copy()
    ttR = np.reshape(ttR, [ttR.shape[1], ttR.shape[2]])
    integral = operations.integrateFunctional(elem, ttR, integrPoints, 0, True, precalc=True)
    integral = integral[np.newaxis, :-1, :]
    ttFx[0] = integral

    ttT = ttFx[1].copy()
    preshape = ttT.shape
    ttT = np.transpose(ttT, [1, 0, 2])
    ttT = np.reshape(ttT, [preshape[1], preshape[0]*preshape[2]])
    integral = operations.integrateFunctional(elem, ttT, integrPoints, 1, True, precalc=True)
    integral = integral[:, :, np.newaxis]
    integral = np.reshape(integral, [elem[1].approxOrder, preshape[0], preshape[2]])
    integral = np.transpose(integral, [1, 0, 2])
    ttFx[1] = integral
    ttP = ttFx[2].copy()
    preshape = ttP.shape
    ttP = np.transpose(ttP, [1, 0, 2])
    ttP = np.reshape(ttP, [preshape[1], preshape[0] * preshape[2]])
    integral = operations.integrateFunctional(elem, ttP, integrPoints, 2, True, precalc=True)
    integral = integral[:, :, np.newaxis]
    integral = np.reshape(integral, [elem[2].approxOrder, preshape[0], preshape[2]])
    integral = np.transpose(integral, [1, 0, 2])
    ttFx[2] = integral
    # r, t, p = elem.getGridList()
    # rr, tt, pp = approx.meshgrid(r[:-1], t, p)
    # grid = approx.meshgrid(r[:-1], t, p)
    # fx = -np.sqrt(rr**2 + tt**2 + pp**2)
    # ttFx = approx.simpleTTsvd(fx)
    # ttfx = approx.alterLeastSquares(ttC, ttFx, np.ones([elem.dim, 2]))
    # print("difference between tensors is: ", np.max(np.abs(approx.toFullTensor(ttFx) - fx)))
    fx = (approx.toFullTensor(ttFx)).flatten()
    # anothertt = approx.simpleTTsvd(np.reshape(C.dot(fx.flatten()), elem.approxOrder - [1, 0, 0]))
    # for it in anothertt:
    #     print("another tt shape", it.shape)
    # print(np.max(np.abs(C.dot(fx.flatten()) - approx.toFullTensor(ttfx).flatten())))
    # time.sleep(500)
    t = time.time()
    # print("everything is calculated, starting system solving")
    # ttSol = sp_lin.solve(C, fx)
    # ttSol = sp_lin.sol
    # print("sparse solver done in ", time.time() - t)
    # print("solving with TT als algorithm")
    # t = time.time()
    ttSol = approx.alterLeastSquares(ttC, ttFx, 7).matricize()
    T = time.time() - t
    # print("ALS solver done in ", time.time() - t)
    # print("difference is ", np.max(np.abs(ttSol - sol)))
    # print("max is: ", np.max(np.abs(sol)))
    # sol = np.reshape(sol, elem.approxOrder - [1, 0, 0])
    # ttFx = approx.simpleTTsvd(sol, tol=1e-6)
    # print("solution tt ranks")
    # for i in ttFx:
    #     print(i.shape)
    r, t, p = elem.getGridList()
    # print(np.cos(2*t))
    rr, tt, pp = approx.meshgrid(r[:-1], t, p)
    grid = approx.meshgrid(r[:-1], t, p)
    fx = -(np.exp(-rr) * (np.sin(pp)) * np.cos(2 * tt) + np.exp(-2*rr) * (np.cos(2*pp)) * np.cos(4 * tt))
    # ttFx = approx.simpleTTsvd(fx, tol=1e-6)
    # print("fx tt ranks")
    # for i in ttFx:
    #     print(i.shape)
    # print("max difference on grid", np.max(np.abs(ttSol - fx.flatten())))
    print(T, np.max(np.abs(ttSol + fx.flatten())))


def solveSphericalPoisVecRHS(polyOrder, vecRHS_F, vecRHS_elem : element,
                             solutionTT_ranks: int, integrPoints=350, ):

    elem = element(np.array([[0, np.inf], [0, np.pi], [0, 2*np.pi]]),
                   np.array(polyOrder, dtype=int), np.array([1, 0, 3]))

    rMatrixD = operations.integrateBilinearForm1_SameElement(elem, lambda x: x * x, integrPoints, 0)[:-1, :-1]
    rMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 0)[:-1, :-1]

    tMatrixD = operations.integrateBilinearForm1_SameElement(elem, lambda x: np.sin(x) ** 2, integrPoints, 1) + \
               operations.integrateBilinearForm2(elem, lambda x: np.sin(x) * np.cos(x), integrPoints, 1)

    tMatrixIr = operations.integrateBilinearForm0(elem, lambda x: np.sin(x)**2, integrPoints, 1)
    tMatrixIp = operations.integrateBilinearForm0(elem, lambda x: x*0 + 1.0, integrPoints, 1)

    pMatrixD = operations.integrateBilinearForm1_SameElement(elem, lambda x: x * 0 + 1.0, integrPoints, 2)
    pMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 2)

    ttC = approx.kronSumtoTT_blockFormat([[None, -rMatrixD, rMatrixI],
                                         [tMatrixIr, -tMatrixD, tMatrixIp],
                                         [pMatrixI, -pMatrixD, None]])

    grid = elem.getGridList()
    w, idNodes = integr.reg_32_wn(-1, 1, integrPoints)
    grid[0] = elem[0].map(idNodes)
    # grid[1] = elem[1].map(idNodes)
    # grid[2] = elem[2].map(idNodes)
    #grid = approx.meshgrid(*grid)

    ttFx = vecRHS_F
    core0 = ttFx[0]
    core0 = np.transpose(core0, [1, 0, 2])
    rhs_grid0 = vecRHS_elem.getGridList()[0]
    core0 = spec.barycentricChebInterpolate(core0, grid[0],
                                            a=rhs_grid0[0], b=rhs_grid0[-1], extrapolation=0, axis=0)
    ttFx[0] = np.transpose(core0, [1, 0, 2])

    grid1 = idNodes
    core1 = ttFx[1]
    core1 = np.transpose(core1, [1, 0, 2])
    core1 = spec.barycentricChebInterpolate(core1, grid1, a=-1, b=1, extrapolation=0, axis=0)
    core1 = np.transpose(core1, [1, 0, 2])
    ttFx[1] = core1
    # print(core1.shape)
    grid2 = idNodes
    core2 = ttFx[2]
    core2 = np.transpose(core2, [1, 0, 2])
    core2 = np.einsum('ij, jnk -> ink', elem[2].eval(grid2), core2)

    core2 = np.transpose(core2, [1, 0, 2])
    ttFx[2] = core2

    # print('cores of f tt decomposition have the following shapes: ')
    # for i in ttFx:
    #     print(i.shape)
    # time.sleep(500)
    ttR = ttFx[0].copy()
    ttR = np.reshape(ttR, [ttR.shape[1], ttR.shape[2]])
    integral = operations.integrateFunctional(elem, ttR, integrPoints, 0, True, precalc=True)
    integral = integral[np.newaxis, :-1, :]
    ttFx[0] = integral

    ttT = ttFx[1].copy()
    preshape = ttT.shape
    ttT = np.transpose(ttT, [1, 0, 2])
    ttT = np.reshape(ttT, [preshape[1], preshape[0]*preshape[2]])
    integral = operations.integrateFunctional(elem, ttT, integrPoints, 1, True, precalc=True)
    integral = integral[:, :, np.newaxis]
    integral = np.reshape(integral, [elem[1].approxOrder, preshape[0], preshape[2]])
    integral = np.transpose(integral, [1, 0, 2])
    ttFx[1] = integral
    ttP = ttFx[2].copy()
    preshape = ttP.shape
    ttP = np.transpose(ttP, [1, 0, 2])
    ttP = np.reshape(ttP, [preshape[1], preshape[0] * preshape[2]])
    integral = operations.integrateFunctional(elem, ttP, integrPoints, 2, True, precalc=True)
    integral = integral[:, :, np.newaxis]
    integral = np.reshape(integral, [elem[2].approxOrder, preshape[0], preshape[2]])
    integral = np.transpose(integral, [1, 0, 2])
    ttFx[2] = integral
    #t = time.time()
    ttSol = approx.alterLeastSquares(ttC, ttFx, solutionTT_ranks)
    #T = time.time() - t
    solCores = []
    for it in ttSol.cores:
        shape = it.shape
        solCores.append(np.reshape(it, [shape[0], shape[1], shape[3]]))
    shape1 = solCores[0].shape
    solCores[0] = np.hstack([solCores[0], np.zeros(shape=(shape1[0], 1, shape1[2]))])
    return solCores


def solveSphericalPoisMatRHS(polyOrder, matRHS_F, matRHS_elem : element,
                             solutionTT_ranks: int,  invRanks: int, integrPoints: int = 350,
                             rounding=False):
    """
            Returns:
                ttSolFull: incides in cores are [r_left, i_n, j_n, r_right]
                where i_n is the value at i's grid point of poisson problem
                      j_n is a function obtained from basis functions product from problem input
        """
    elem = element(np.array([[0, np.inf], [0, np.pi], [0, 2*np.pi]]),
                   np.array(polyOrder, dtype=int), np.array([1, 0, 3]))

    rMatrixD = operations.integrateBilinearForm1_SameElement(elem, lambda x: x * x, integrPoints, 0)[:-1, :-1]
    rMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 0)[:-1, :-1]

    tMatrixD = operations.integrateBilinearForm1_SameElement(elem, lambda x: np.sin(x) ** 2, integrPoints, 1) + \
               operations.integrateBilinearForm2(elem, lambda x: np.sin(x) * np.cos(x), integrPoints, 1)

    tMatrixIr = operations.integrateBilinearForm0(elem, lambda x: np.sin(x)**2, integrPoints, 1)
    tMatrixIp = operations.integrateBilinearForm0(elem, lambda x: x*0 + 1.0, integrPoints, 1)

    pMatrixD = operations.integrateBilinearForm1_SameElement(elem, lambda x: x * 0 + 1.0, integrPoints, 2)
    pMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 2)

    ttC = approx.kronSumtoTT_blockFormat([[None, -rMatrixD, rMatrixI],
                                         [tMatrixIr, -tMatrixD, tMatrixIp],
                                         [pMatrixI, -pMatrixD, None]])
    inv_ttC = TT(approx.invertTT_Matrix(ttC, invRanks), rounding)

    w, idNodes = integr.reg_32_wn(-1, 1, integrPoints)


    ttFx = matRHS_F
    rhs_grid0 = matRHS_elem.getGridList()[0]

    ttFx[0] = spec.barycentricChebInterpolate(ttFx[0], elem[0].map(idNodes),
                                a=rhs_grid0[0], b=rhs_grid0[-1], extrapolation=0, axis=0)

    ttFx[0] = operations.integrateFunctionalWithMatrixRHS(elem,
                        ttFx[0], integrPoints, 0, lambdaWeightAlongAxis=lambda x: x * x)[:-1, :, :]

    ttFx[1] = spec.barycentricChebInterpolate(ttFx[1], idNodes, a=-1, b=1, extrapolation=0, axis=0)
    ttFx[1] = operations.integrateFunctionalWithMatrixRHS(elem,
                        ttFx[1], integrPoints, 1, lambdaWeightAlongAxis=lambda x: x * x)

    ttFx[2] = np.einsum('ij, jk -> ik', elem[2].eval(idNodes), ttFx[2])
    ttFx[2] = operations.integrateFunctionalWithMatrixRHS(elem,
                        ttFx[2], integrPoints, 2)


    smallTT = []
    for i in range(len(polyOrder)):
        shape = ttFx[i].shape
        core = np.reshape(ttFx[i], [shape[0], shape[1]*shape[2]])
        smallTT.append(core[np.newaxis, :, :, np.newaxis])


    ttSolFull = inv_ttC.dot(TT(smallTT)).cores
    shape = ttSolFull[0].shape
    zeros = np.zeros([shape[0], shape[1] + 1, shape[2], shape[3]])
    zeros[:, :-1, :, :] = ttSolFull[0]
    ttSolFull[0] = zeros
    return ttSolFull



def solveSphericalPoisNoPhi(polyOrder, rhsF, integrPoints=2000):

    elem = element(np.array([[0, np.inf], [0, np.pi]]),
                   np.array(polyOrder, dtype=int), np.array([1, 0]))

    rMatrixD = operations.integrateBilinearForm1_SameElement(elem, lambda x: x * x, integrPoints, 0)[:-1, :-1]
    rMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 0)[:-1, :-1]

    # tMatrixD = operations.integrateBilinearForm1(elem, lambda x: np.sin(x), integrPoints, 1)
    tMatrixD = operations.integrateBilinearForm1_SameElement(elem, lambda x: np.sin(x) ** 2, integrPoints, 1) + \
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


def testSphericalLaplacianInverseTT(polyOrder, integrPoints=500):
    elem = element(np.array([[0, np.inf], [0, np.pi], [0, 2 * np.pi]]),
                   np.array(polyOrder, dtype=int), np.array([1, 0, 3]))

    rMatrixD = operations.integrateBilinearForm1_SameElement(elem, lambda x: x * x, integrPoints, 0)[:-1, :-1]
    rMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 0)[:-1, :-1]

    tMatrixD = operations.integrateBilinearForm1_SameElement(elem, lambda x: np.sin(x) ** 2, integrPoints, 1) + \
               operations.integrateBilinearForm2(elem, lambda x: np.sin(x) * np.cos(x), integrPoints, 1)

    tMatrixIr = operations.integrateBilinearForm0(elem, lambda x: np.sin(x) ** 2, integrPoints, 1)
    tMatrixIp = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 1)

    pMatrixD = operations.integrateBilinearForm1_SameElement(elem, lambda x: x * 0 + 1.0, integrPoints, 2)
    pMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 2)

    ttC = approx.kronSumtoTT_blockFormat([[None, -rMatrixD, rMatrixI],
                                          [tMatrixIr, -tMatrixD, tMatrixIp],
                                          [pMatrixI, -pMatrixD, None]])
    truePolyOrder = np.array(polyOrder) - np.array([1, 0, 0])
    #C = (approx.toFullTensor(ttC, True))
    #invC = sp_lin.inv(np.reshape(C, [np.prod(truePolyOrder), np.prod(truePolyOrder)]))
    shape = polyOrder
    #ttC = approx.matrixTTsvd(C, truePolyOrder, tol=1e-6)
    ttInvC = approx.invertTT_Matrix(ttC, 10)
    #ttInvC = approx.matrixTTsvd(invC, truePolyOrder, tol=1e-6)
    #print("nonzero amount in kron matrix: ", np.count_nonzero(C))
    #print("nonzero amount in inverse matrix: ", np.count_nonzero(invC))
    sum = 0
    for i in range(len(ttC)):
        print("C core shape:", ttC[i].shape, "inverse C core shape ", ttInvC[i].shape, "with the total size"
                                                                                       "of ", ttInvC[i].size)
        sum += ttInvC[i].size
    #print("inverse and TT approximation nonzero elements magnitude difference (times):", np.count_nonzero(invC) / sum)
    for i in range(3):
        shape = ttInvC[i].shape
        ttInvC[i] = np.reshape(ttInvC[i], [shape[0], shape[1] * shape[2], shape[3]])

        shape = ttC[i].shape
        ttC[i] = np.reshape(ttC[i], [shape[0], shape[1] * shape[2], shape[3]])

    # newCshape = np.array(truePolyOrder, dtype=int)
    # invC = np.reshape(invC, [*newCshape, *newCshape])
    # invC = np.transpose(invC, axes=[0, 3, 1, 4, 2, 5])
    # invC = np.reshape(invC, newCshape ** 2)
    # A = np.reshape(ttInvC[0], [ttInvC[0].shape[1], ttInvC[0].shape[-1]])
    # initShape = invC.shape
    # for i in range(1, len(ttInvC)):
    #     shape = ttInvC[i].shape
    #     reshaped = np.reshape(ttInvC[i], [shape[0], shape[1] * shape[2]])
    #     A = np.dot(A, reshaped)
    #     A = np.reshape(A, [np.prod(initShape[:i + 1]), int(A.shape[-1] / initShape[i])])
    # A = np.reshape(A, initShape)
    # print("approximation error for inverse: ", np.sum((A - invC) ** 2))


#testSphericalLaplacianInverseTT([30, 8, 8])

# def solveSphericalPoisForPlot(iStart, iEnd, iStep):
#     for i in range(iStart, iEnd, iStep):
#         solveSphericalPois(np.array([i, i, min(i, 14)]), lambda x: np.exp(-x[0])*(np.sin(x[2])*(np.cos(2*x[1])*(-1.0 + (-4.0 + (-2.0 + x[0])*x[0])*(np.sin(x[1])**2)) - (np.sin(2*x[1])**2)) +
#                                           np.exp(-x[0])*4*np.cos(2*x[2])*(np.cos(4*x[1])*(-1.0 + (-4.0 + (-1.0 + x[0])*x[0])*(np.sin(x[1])**2)) - \
#                                                                     np.cos(x[1])*np.sin(x[1])*np.sin(4*x[1])))
#                        , integrPoints=500)