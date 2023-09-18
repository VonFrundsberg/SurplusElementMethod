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
import scikit_tt.tensor_train as tt
from scikit_tt.tensor_train import TT
from heap import solveSphericalPoisMatRHS as poissonSolution
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
import time
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


    C = sp_sparse.kron(rMatrixD, tMatrixI)
    C += sp_sparse.kron(rMatrixI, tMatrixD)
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

def solveEigenSphericalPois(polyOrder, potential, integrPoints=350):

    elem = element(np.array([[0, 20], [0, np.pi], [0, 2*np.pi]]),
                   np.array(polyOrder, dtype=int), np.array([0, 0, 3]))

    rMatrixD = operations.integrateBilinearForm1(elem, lambda x: x * x, integrPoints, 0)[:-1, :-1]
    rMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 0)[:-1, :-1]

    tMatrixD = operations.integrateBilinearForm1(elem, lambda x: np.sin(x)**2, integrPoints, 1) + \
               operations.integrateBilinearForm2(elem, lambda x: np.sin(x)*np.cos(x), integrPoints, 1)

    tMatrixIr = operations.integrateBilinearForm0(elem, lambda x: np.sin(x)**2, integrPoints, 1)
    tMatrixIp = operations.integrateBilinearForm0(elem, lambda x: x*0 + 1.0, integrPoints, 1)

    pMatrixD = operations.integrateBilinearForm1(elem, lambda x: x * 0 + 1.0, integrPoints, 2)
    pMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 2)

    ttA = approx.kronSumtoTT_blockFormat([[None, rMatrixD, rMatrixI],
                                         [tMatrixIr, tMatrixD, tMatrixIp],
                                         [pMatrixI, pMatrixD, None]])

    rMatrixI = operations.integrateBilinearForm0(elem, lambda x: x, integrPoints, 0)[:-1, :-1]
    tMatrixI = operations.integrateBilinearForm0(elem, lambda x: np.sin(x) ** 2, integrPoints, 1)
    pMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 2)
    ttV = [2*rMatrixI, tMatrixI, pMatrixI]
    for i in range(len(ttV)):
        ttV[i] = -ttV[i][np.newaxis, :, :, np.newaxis]

    rMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * x, integrPoints, 0)[:-1, :-1]
    tMatrixI = operations.integrateBilinearForm0(elem, lambda x: np.sin(x) ** 2, integrPoints, 1)
    pMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 2)

    ttB = [2*rMatrixI, tMatrixI, pMatrixI]
    for i in range(len(ttB)):
        ttB[i] = ttB[i][np.newaxis, : , :, np.newaxis]

    #print("solving with TT als algorithm")
    t = time.time()
    ttSol = approx.eigAlterLeastSquares(A=ttA, B=ttB,  ranks=4, sigma=-0.5, V=ttV, real=True)
    eigsList = []
    eigVecList = []
    eigVecList.append(ttSol[1])
    eigsList.append(ttSol[0])
    # for i in range(4):
    #     ttSol = approx.eigAlterLeastSquares(A=ttA, B=ttB, ranks=6, sigma=-0.5,
    #                                         V=ttV, prev=eigVecList, real=True, shift=eigsList[-1])
    #     eigVecList.append(ttSol[1])
    #     eigsList.append(ttSol[0])
    print(eigsList)


def solveEigenSphericalPois6dDifferentOrder(polyOrder, integrPoints=350):

    elem = element(np.array([[0, 20], [0, np.pi], [0, 2*np.pi]]),
                   np.array(polyOrder, dtype=int), np.array([0, 0, 3]))

    rMatrixD = operations.integrateBilinearForm1(elem, lambda x: x * x, integrPoints, 0)[:-1, :-1]
    rMatrixI1 = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 0)[:-1, :-1]
    rMatrixI2 = operations.integrateBilinearForm0(elem, lambda x: x * x, integrPoints, 0)[:-1, :-1]

    tMatrixD = operations.integrateBilinearForm1(elem, lambda x: np.sin(x)**2, integrPoints, 1) + \
               operations.integrateBilinearForm2(elem, lambda x: np.sin(x)*np.cos(x), integrPoints, 1)

    tMatrixIr = operations.integrateBilinearForm0(elem, lambda x: np.sin(x)**2, integrPoints, 1)
    tMatrixIp = operations.integrateBilinearForm0(elem, lambda x: x*0 + 1.0, integrPoints, 1)

    pMatrixD = operations.integrateBilinearForm1(elem, lambda x: x * 0 + 1.0, integrPoints, 2)
    pMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 2)

    ttA = approx.sphericalLaplace6d([[rMatrixI1, rMatrixD, rMatrixI2],
                                    [tMatrixIr, tMatrixD, tMatrixIp],
                                    [pMatrixI, pMatrixD, None]
                                          ])
    #approx.vecRound(ttA.cores, 1e-6, True)
    #approx.printTT(ttA.cores)
    #time.sleep(500)
    # ttA1.append(rMatrixI2[np.newaxis, :, :, np.newaxis])
    # ttA1.append(tMatrixIr[np.newaxis, :, :, np.newaxis])
    # ttA1.append(pMatrixI[np.newaxis, :, :, np.newaxis])

    # ttA2 = approx.kronSumtoTT_blockFormat([[None, rMatrixD, rMatrixI1],
    #                                        [tMatrixIr, tMatrixD, tMatrixIp],
    #                                        [pMatrixI, pMatrixD, None],
    #                                        ])
    # ttA2.insert(0, rMatrixI2[np.newaxis, :, :, np.newaxis])
    # ttA2.insert(1, tMatrixIr[np.newaxis, :, :, np.newaxis])
    # ttA2.insert(2, pMatrixI[np.newaxis, :, :, np.newaxis])

    rMatrixD = None; rMatrixI = None; tMatrixD = None; tMatrixIr = None; tMatrixIp = None;
    pMatrixD = None; pMatrixI = None;
    Z = 2
    rMatrixI1 = operations.integrateBilinearForm0(elem, lambda x: -Z*2*x, integrPoints, 0)[:-1, :-1]
    rMatrixI2 = operations.integrateBilinearForm0(elem, lambda x: x * x, integrPoints, 0)[:-1, :-1]
    tMatrixI = operations.integrateBilinearForm0(elem, lambda x: np.sin(x) ** 2, integrPoints, 1)
    pMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 2)
    # ttV1 = [rMatrixI1, tMatrixI, pMatrixI, rMatrixI2, tMatrixI, pMatrixI]
    # ttV2 = [rMatrixI2, tMatrixI, pMatrixI, rMatrixI1, tMatrixI, pMatrixI]

    ttV1 = [rMatrixI1, rMatrixI2, tMatrixI, tMatrixI, pMatrixI, pMatrixI]
    ttV2 = [rMatrixI2, rMatrixI1, tMatrixI, tMatrixI, pMatrixI, pMatrixI]
    for i in range(len(ttV1)):
        ttV1[i] = ttV1[i][np.newaxis, :, :, np.newaxis]
        ttV2[i] = ttV2[i][np.newaxis, :, :, np.newaxis]

    rMatrixI1 = None; rMatrixI2 = None
    tMatrixI = None;  pMatrixI = None;

    rMatrixI = operations.integrateBilinearForm0(elem, lambda x: np.sqrt(2) * x * x, integrPoints, 0)[:-1, :-1]

    tMatrixI = operations.integrateBilinearForm0(elem, lambda x: np.sin(x) ** 2, integrPoints, 1)

    pMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 2)

    ttB = [rMatrixI, rMatrixI, tMatrixI, tMatrixI, pMatrixI, pMatrixI]
    for i in range(len(ttB)):
        ttB[i] = ttB[i][np.newaxis, : , :, np.newaxis]

    rMatrixI = None;    tMatrixI = None;    pMatrixI = None;

    #print("solving with TT als algorithm")
    t = time.time()
    ttSol = approx.eigAlterLeastSquares2d(A=ttA, B=ttB, ranks=1, sigma=-10, real=True, V=[ttV1, ttV2])
    print("init energy: " , ttSol[0])
    solCores = []
    for it in ttSol[1].cores:
        shape = it.shape
        solCores.append(np.reshape(it, [shape[0], shape[1], shape[3]]))
    shape1 = solCores[0].shape
    shape2 = solCores[3].shape
    solCores[0] = np.hstack([solCores[0], np.zeros(shape=(shape1[0], 1, shape1[2]))])
    solCores[3] = np.hstack([solCores[3], np.zeros(shape=(shape2[0], 1, shape2[2]))])
    return solCores
def solveEigenSphericalPois6d(polyOrder, integrPoints=350):

    elem = element(np.array([[0, 20], [0, np.pi], [0, 2*np.pi]]),
                   np.array(polyOrder, dtype=int), np.array([0, 0, 3]))

    rMatrixD = operations.integrateBilinearForm1(elem, lambda x: x * x, integrPoints, 0)[:-1, :-1]
    rMatrixI1 = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 0)[:-1, :-1]
    rMatrixI2 = operations.integrateBilinearForm0(elem, lambda x: x * x, integrPoints, 0)[:-1, :-1]

    tMatrixD = operations.integrateBilinearForm1(elem, lambda x: np.sin(x)**2, integrPoints, 1) + \
               operations.integrateBilinearForm2(elem, lambda x: np.sin(x)*np.cos(x), integrPoints, 1)

    tMatrixIr = operations.integrateBilinearForm0(elem, lambda x: np.sin(x)**2, integrPoints, 1)
    tMatrixIp = operations.integrateBilinearForm0(elem, lambda x: x*0 + 1.0, integrPoints, 1)

    pMatrixD = operations.integrateBilinearForm1(elem, lambda x: x * 0 + 1.0, integrPoints, 2)
    pMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 2)

    ttA1 = approx.kronSumtoTT_blockFormat([[None, rMatrixD, rMatrixI1],
                                          [tMatrixIr, tMatrixD, tMatrixIp],
                                          [pMatrixI, pMatrixD, None],
                                          ])

    ttA1.append(rMatrixI2[np.newaxis, :, :, np.newaxis])
    ttA1.append(tMatrixIr[np.newaxis, :, :, np.newaxis])
    ttA1.append(pMatrixI[np.newaxis, :, :, np.newaxis])

    ttA2 = approx.kronSumtoTT_blockFormat([[None, rMatrixD, rMatrixI1],
                                           [tMatrixIr, tMatrixD, tMatrixIp],
                                           [pMatrixI, pMatrixD, None],
                                           ])
    ttA2.insert(0, rMatrixI2[np.newaxis, :, :, np.newaxis])
    ttA2.insert(1, tMatrixIr[np.newaxis, :, :, np.newaxis])
    ttA2.insert(2, pMatrixI[np.newaxis, :, :, np.newaxis])

    rMatrixD = None; rMatrixI = None; tMatrixD = None; tMatrixIr = None; tMatrixIp = None;
    pMatrixD = None; pMatrixI = None;
    Z = 2
    rMatrixI1 = operations.integrateBilinearForm0(elem, lambda x: -Z*2*x, integrPoints, 0)[:-1, :-1]
    rMatrixI2 = operations.integrateBilinearForm0(elem, lambda x: x * x, integrPoints, 0)[:-1, :-1]
    tMatrixI = operations.integrateBilinearForm0(elem, lambda x: np.sin(x) ** 2, integrPoints, 1)
    pMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 2)
    ttV1 = [rMatrixI1, tMatrixI, pMatrixI, rMatrixI2, tMatrixI, pMatrixI]
    ttV2 = [rMatrixI2, tMatrixI, pMatrixI, rMatrixI1, tMatrixI, pMatrixI]
    for i in range(len(ttV1)):
        ttV1[i] = ttV1[i][np.newaxis, :, :, np.newaxis]
        ttV2[i] = ttV2[i][np.newaxis, :, :, np.newaxis]

    rMatrixI1 = None; rMatrixI2 = None
    tMatrixI = None;  pMatrixI = None;

    rMatrixI = operations.integrateBilinearForm0(elem, lambda x: np.sqrt(2) * x * x, integrPoints, 0)[:-1, :-1]

    tMatrixI = operations.integrateBilinearForm0(elem, lambda x: np.sin(x) ** 2, integrPoints, 1)

    pMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 2)

    ttB = [rMatrixI, tMatrixI, pMatrixI, rMatrixI, tMatrixI, pMatrixI]
    for i in range(len(ttB)):
        ttB[i] = ttB[i][np.newaxis, : , :, np.newaxis]

    rMatrixI = None;    tMatrixI = None;    pMatrixI = None;

    #print("solving with TT als algorithm")
    t = time.time()
    ttSol = approx.eigAlterLeastSquares2d(A=[ttA1, ttA2], B=ttB, ranks=1, sigma=-10, real=True, V=[ttV1, ttV2])
    print("init energy: " , ttSol[0])
    solCores = []
    for it in ttSol[1].cores:
        shape = it.shape
        solCores.append(np.reshape(it, [shape[0], shape[1], shape[3]]))
    shape1 = solCores[0].shape
    shape2 = solCores[3].shape
    solCores[0] = np.hstack([solCores[0], np.zeros(shape=(shape1[0], 1, shape1[2]))])
    solCores[3] = np.hstack([solCores[3], np.zeros(shape=(shape2[0], 1, shape2[2]))])
    return solCores


def solveEigenSphericalPois6dRepulsionOld(polyOrder, prevSolutionCores, integrPoints=350):

    elem = element(np.array([[0, 20], [0, np.pi], [0, 2*np.pi]]),
                   np.array(polyOrder, dtype=int), np.array([0, 0, 3]))

    rMatrixD = operations.integrateBilinearForm1(elem, lambda x: x * x, integrPoints, 0)[:-1, :-1]
    rMatrixI1 = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 0)[:-1, :-1]
    rMatrixI2 = operations.integrateBilinearForm0(elem, lambda x: x * x, integrPoints, 0)[:-1, :-1]

    tMatrixD = operations.integrateBilinearForm1(elem, lambda x: np.sin(x)**2, integrPoints, 1) + \
               operations.integrateBilinearForm2(elem, lambda x: np.sin(x)*np.cos(x), integrPoints, 1)

    tMatrixIr = operations.integrateBilinearForm0(elem, lambda x: np.sin(x)**2, integrPoints, 1)
    tMatrixIp = operations.integrateBilinearForm0(elem, lambda x: x*0 + 1.0, integrPoints, 1)

    pMatrixD = operations.integrateBilinearForm1(elem, lambda x: x * 0 + 1.0, integrPoints, 2)
    pMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 2)

    ttA1 = approx.kronSumtoTT_blockFormat([[None, rMatrixD, rMatrixI1],
                                          [tMatrixIr, tMatrixD, tMatrixIp],
                                          [pMatrixI, pMatrixD, None],
                                          ])

    ttA1.append(rMatrixI2[np.newaxis, :, :, np.newaxis])
    ttA1.append(tMatrixIr[np.newaxis, :, :, np.newaxis])
    ttA1.append(pMatrixI[np.newaxis, :, :, np.newaxis])

    ttA2 = approx.kronSumtoTT_blockFormat([[None, rMatrixD, rMatrixI1],
                                           [tMatrixIr, tMatrixD, tMatrixIp],
                                           [pMatrixI, pMatrixD, None],
                                           ])
    ttA2.insert(0, rMatrixI2[np.newaxis, :, :, np.newaxis])
    ttA2.insert(1, tMatrixIr[np.newaxis, :, :, np.newaxis])
    ttA2.insert(2, pMatrixI[np.newaxis, :, :, np.newaxis])

    rMatrixD = None; rMatrixI = None; tMatrixD = None; tMatrixIr = None; tMatrixIp = None;
    pMatrixD = None; pMatrixI = None;
    for it in prevSolutionCores:
        print(it.shape)
    time.sleep(500)
    potentialY = poissonSolution(polyOrder, tt.eye, elem,
                                 integrPoints=integrPoints, solutionTT_ranks=10)

    poissonElem = element(np.array([[0, np.inf], [0, np.pi], [0, 2 * np.pi]]),
                   np.array(polyOrder, dtype=int), np.array([1, 0, 3]))
    #poisGrid = poissonElem[0].getMappedRefPoints()
    #print("poisson grid", poisGrid)
    refGrid = elem[0].getMappedRefPoints()
    #print("reference grid", refGrid)
    #print("reference grid to poisson grid", poissonElem[0].inverseMap(refGrid))
    refGridToPoisGridUnit = poissonElem[0].inverseMap(refGrid)
    rShapeComponentOfY = potentialY[0].shape
    rComponentOfY = np.reshape(potentialY[0], [rShapeComponentOfY[1], rShapeComponentOfY[2]])
    rComponentOfY = spec.barycentricChebInterpolate(rComponentOfY, x=refGridToPoisGridUnit, a=-1, b=1)
    potentialY[0] = rComponentOfY[np.newaxis, :, :]
    Y = approx.hadamardProduct(potentialY[:3], prevSolutionCores[3:])
    # for it in Y:
    #     print(it.shape)

    grid = elem.getGridList()
    w, idNodes = integr.reg_32_wn(-1, 1, integrPoints)

    ttFx = Y
    grid0 = idNodes
    core0 = ttFx[0]
    core0 = np.transpose(core0, [1, 0, 2])
    core0 = spec.barycentricChebInterpolate(core0, grid0, a=-1, b=1, extrapolation=0, axis=0)

    core0 = np.transpose(core0, [1, 0, 2])
    ttFx[0] = core0 * (grid0[np.newaxis, : , np.newaxis])**2

    grid1 = idNodes
    core1 = ttFx[1]
    core1 = np.transpose(core1, [1, 0, 2])
    core1 = spec.barycentricChebInterpolate(core1, grid1, a=-1, b=1, extrapolation=0, axis=0)
    core1 = np.transpose(core1, [1, 0, 2])
    ttFx[1] = core1 * (grid1[np.newaxis, : , np.newaxis])**2
    # print(core1.shape)
    grid2 = idNodes
    core2 = ttFx[2]
    core2 = np.transpose(core2, [1, 0, 2])
    core2 = np.einsum('ij, jnk -> ink', elem[2].eval(grid2), core2)

    core2 = np.transpose(core2, [1, 0, 2])
    ttFx[2] = core2

    integralOfY = np.squeeze(approx.integrateTT(ttFx, [w, w, w]))
    ttFx = None
    core1 = None; core2 = None; core0 = None; grid0 = None; grid1 = None; grid2 = None

    Z = 2
    rMatrixI1 = operations.integrateBilinearForm0(elem, lambda x: -Z*2*x, integrPoints, 0)[:-1, :-1]
    rMatrixI2 = operations.integrateBilinearForm0(elem, lambda x: x * x, integrPoints, 0)[:-1, :-1]
    tMatrixI = operations.integrateBilinearForm0(elem, lambda x: np.sin(x) ** 2, integrPoints, 1)
    pMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 2)
    ttV1 = [rMatrixI1, tMatrixI, pMatrixI, rMatrixI2, tMatrixI, pMatrixI]
    ttV2 = [rMatrixI2, tMatrixI, pMatrixI, rMatrixI1, tMatrixI, pMatrixI]
    arrayY = []
    for i in range(len(ttV1)):
        ttV1[i] = ttV1[i][np.newaxis, :, :, np.newaxis]
        ttV2[i] = ttV2[i][np.newaxis, :, :, np.newaxis]
        arrayY.append(np.ones(ttV1[i].shape))
    integralOfY = -1
    arrayY[0] *= np.pi*2*integralOfY

    rMatrixI1 = None; rMatrixI2 = None
    tMatrixI = None;  pMatrixI = None;

    rMatrixI = operations.integrateBilinearForm0(elem, lambda x: np.sqrt(2) * x * x, integrPoints, 0)[:-1, :-1]

    tMatrixI = operations.integrateBilinearForm0(elem, lambda x: np.sin(x) ** 2, integrPoints, 1)

    pMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 2)

    ttB = [rMatrixI, tMatrixI, pMatrixI, rMatrixI, tMatrixI, pMatrixI]
    for i in range(len(ttB)):
        ttB[i] = ttB[i][np.newaxis, : , :, np.newaxis]

    rMatrixI = None;    tMatrixI = None;    pMatrixI = None;

    #print("solving with TT als algorithm")
    t = time.time()

    ttSol = approx.eigAlterLeastSquares2d(A=[ttA1, ttA2], B=[ttB, arrayY],
                                          ranks=1, sigma=-10, real=True, V=[ttV1, ttV2])
    print("post energy: " , ttSol[0])
    solCores = []
    for it in ttSol[1].cores:
        shape = it.shape
        solCores.append(np.reshape(it, [shape[0], shape[1], shape[3]]))
    shape1 = solCores[0].shape
    shape2 = solCores[3].shape
    solCores[0] = np.hstack([solCores[0], np.zeros(shape=(shape1[0], 1, shape1[2]))])
    solCores[3] = np.hstack([solCores[3], np.zeros(shape=(shape2[0], 1, shape2[2]))])
    return solCores



def solveEigenSphericalPois6dRepulsion(polyOrder, integrPoints=350):

    elem = element(np.array([[0, 20], [0, np.pi], [0, 2*np.pi]]),
                   np.array(polyOrder, dtype=int), np.array([0, 0, 3]))

    rMatrixD = operations.integrateBilinearForm1(elem, lambda x: x * x, integrPoints, 0)[:-1, :-1]
    rMatrixI1 = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 0)[:-1, :-1]
    rMatrixI2 = operations.integrateBilinearForm0(elem, lambda x: x * x, integrPoints, 0)[:-1, :-1]

    tMatrixD = operations.integrateBilinearForm1(elem, lambda x: np.sin(x)**2, integrPoints, 1) + \
               operations.integrateBilinearForm2(elem, lambda x: np.sin(x)*np.cos(x), integrPoints, 1)

    tMatrixIr = operations.integrateBilinearForm0(elem, lambda x: np.sin(x)**2, integrPoints, 1)
    tMatrixIp = operations.integrateBilinearForm0(elem, lambda x: x*0 + 1.0, integrPoints, 1)

    pMatrixD = operations.integrateBilinearForm1(elem, lambda x: x * 0 + 1.0, integrPoints, 2)
    pMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 2)

    ttA1 = approx.kronSumtoTT_blockFormat([[None, rMatrixD, rMatrixI1],
                                          [tMatrixIr, tMatrixD, tMatrixIp],
                                          [pMatrixI, pMatrixD, None],
                                          ])

    ttA1.append(rMatrixI2[np.newaxis, :, :, np.newaxis])
    ttA1.append(tMatrixIr[np.newaxis, :, :, np.newaxis])
    ttA1.append(pMatrixI[np.newaxis, :, :, np.newaxis])

    ttA2 = approx.kronSumtoTT_blockFormat([[None, rMatrixD, rMatrixI1],
                                           [tMatrixIr, tMatrixD, tMatrixIp],
                                           [pMatrixI, pMatrixD, None],
                                           ])
    ttA2.insert(0, rMatrixI2[np.newaxis, :, :, np.newaxis])
    ttA2.insert(1, tMatrixIr[np.newaxis, :, :, np.newaxis])
    ttA2.insert(2, pMatrixI[np.newaxis, :, :, np.newaxis])

    del rMatrixD, rMatrixI1, rMatrixI2
    del tMatrixD, tMatrixIr, tMatrixIp
    del pMatrixD, pMatrixI

    IdCores = []
    for i in range(3):
        IdCores.append(np.eye(polyOrder[i]))
    r12 = lambda x: (x[0]**2 + x[3]**2 - 2*x[0]*x[3]*
                     (np.sin(x[1]*np.sin(x[4])*np.cos(x[2] - x[5]) + np.cos(x[1])*np.cos(x[4]))))
    gridsList = [None]*6
    for i in range(3):
        gridsList[2*i] = elem.getGridList()[i]
        gridsList[2*i + 1] = elem.getGridList()[i]
    grid = approx.meshgrid(*gridsList)
    fr12 = r12(grid)
    del grid
    potentialY = approx.simpleTTsvd(fr12, tol=1e-3)
    approx.printTT(potentialY)
    time.sleep(500)
    poissonElem = element(np.array([[0, np.inf], [0, np.pi], [0, 2 * np.pi]]),
                   np.array(polyOrder, dtype=int), np.array([1, 0, 3]))


    refGridR = elem[0].getMappedRefPoints()
    refGridToPoisGridUnit = poissonElem[0].inverseMap(refGridR)
    rShapeComponentOfY = potentialY[0].shape
    rComponentOfY = np.reshape(potentialY[0],
                    [rShapeComponentOfY[1], rShapeComponentOfY[2] * rShapeComponentOfY[3]])
    rComponentOfY = spec.barycentricChebInterpolate(rComponentOfY, x=refGridToPoisGridUnit, a=-1, b=1, axis=0)
    potentialY[0] = np.reshape(rComponentOfY,
                    [1, rShapeComponentOfY[1], rShapeComponentOfY[2], rShapeComponentOfY[3]])
    del rComponentOfY


    w, idNodes = integr.reg_32_wn(-1, 1, integrPoints)

    potentialY[0] = spec.barycentricChebInterpolateTensorAlongAxis(
        potentialY[0], idNodes, a=-1, b=1, axis=1)

    potentialY[1] = spec.barycentricChebInterpolateTensorAlongAxis(
        potentialY[1], idNodes, a=-1, b=1, axis=1)

    potentialY[2] = np.einsum('ij, ajkb -> aikb', elem[2].eval(idNodes), potentialY[2])

    approx.printTT(potentialY)
    ttY = [None]*6
    for i in range(3):
        if i < 2:
            ttY[i] = operations.integrateBilinearForm0_TensorWeight(
                elem, potentialY[i], integrPoints, i, lambda x: x * x)
        else:
            ttY[i] = operations.integrateBilinearForm0_TensorWeight(
                elem, potentialY[i], integrPoints, i)
    for i in range(3):
        shape = ttY[i].shape
        ttY[i] = np.reshape(ttY[i], [shape[0]*shape[1], shape[2]*shape[3]*shape[4]])
        u, s, v = sp_lin.svd(ttY[i], full_matrices=False)
        u = np.reshape(u[:, 0], [1, shape[0], shape[1], 1])
        v = np.reshape(v[0, :]*s[0], [shape[2], shape[3], shape[4]])
        v = np.transpose(v, [1, 0, 2])
        v = np.reshape(v, [shape[0], shape[1], shape[2], shape[4]])
        v = np.transpose(v, [2, 0, 1, 3])
        if i == 0:
            ttY[i] = -4*np.pi*v[:, :-1, :-1, :]
            ttY[i + 3] = u[:, :-1, :-1, :]
        else:
            ttY[i] = v
            ttY[i + 3] = u
        # print("for ", i, "s index the shapes are")
        # print(u.shape)
        # print(v.shape)
        #
        # cumsum = np.cumsum(s)
        # r_delta = np.argmax(cumsum[-1] - cumsum < 1e-12) + 1
        # print("the rank is ", r_delta)
    #approx.printTT(ttY)
    ttFx = None
    core1 = None; core2 = None; core0 = None; grid0 = None; grid1 = None; grid2 = None

    Z = 2
    rMatrixI1 = operations.integrateBilinearForm0(elem, lambda x: -Z*2*x, integrPoints, 0)[:-1, :-1]
    rMatrixI2 = operations.integrateBilinearForm0(elem, lambda x: x * x, integrPoints, 0)[:-1, :-1]
    tMatrixI = operations.integrateBilinearForm0(elem, lambda x: np.sin(x) ** 2, integrPoints, 1)
    pMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 2)
    ttV1 = [rMatrixI1, tMatrixI, pMatrixI, rMatrixI2, tMatrixI, pMatrixI]
    ttV2 = [rMatrixI2, tMatrixI, pMatrixI, rMatrixI1, tMatrixI, pMatrixI]

    for i in range(len(ttV1)):
        ttV1[i] = ttV1[i][np.newaxis, :, :, np.newaxis]
        ttV2[i] = ttV2[i][np.newaxis, :, :, np.newaxis]

    rMatrixI1 = None; rMatrixI2 = None
    tMatrixI = None;  pMatrixI = None;

    rMatrixI = operations.integrateBilinearForm0(elem, lambda x: np.sqrt(2) * x * x, integrPoints, 0)[:-1, :-1]

    tMatrixI = operations.integrateBilinearForm0(elem, lambda x: np.sin(x) ** 2, integrPoints, 1)

    pMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 2)

    ttB = [rMatrixI, tMatrixI, pMatrixI, rMatrixI, tMatrixI, pMatrixI]
    for i in range(len(ttB)):
        ttB[i] = ttB[i][np.newaxis, : , :, np.newaxis]

    rMatrixI = None;    tMatrixI = None;    pMatrixI = None;

    #print("solving with TT als algorithm")
    t = time.time()

    ttSol = approx.eigAlterLeastSquares2d(A=[ttA1, ttA2], B=ttB,
                                          ranks=2, sigma=-10, real=True, V=[ttV1, ttV2, ttY])
    print("post energy: " , ttSol[0])



def solveEigenSphericalPois6dRepulsionDifferentOrder(polyOrder, integrPoints=350):

    elem = element(np.array([[0, 20], [0, np.pi], [0, 2*np.pi]]),
                   np.array(polyOrder, dtype=int), np.array([0, 0, 3]))

    rMatrixD = operations.integrateBilinearForm1(elem, lambda x: x * x, integrPoints, 0)[:-1, :-1]
    rMatrixI1 = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 0)[:-1, :-1]
    rMatrixI2 = operations.integrateBilinearForm0(elem, lambda x: x * x, integrPoints, 0)[:-1, :-1]

    tMatrixD = operations.integrateBilinearForm1(elem, lambda x: np.sin(x)**2, integrPoints, 1) + \
               operations.integrateBilinearForm2(elem, lambda x: np.sin(x)*np.cos(x), integrPoints, 1)

    tMatrixIr = operations.integrateBilinearForm0(elem, lambda x: np.sin(x)**2, integrPoints, 1)
    tMatrixIp = operations.integrateBilinearForm0(elem, lambda x: x*0 + 1.0, integrPoints, 1)

    pMatrixD = operations.integrateBilinearForm1(elem, lambda x: x * 0 + 1.0, integrPoints, 2)
    pMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 2)

    ttA = approx.sphericalLaplace6d([[rMatrixI1, rMatrixD, rMatrixI2],
                                     [tMatrixIr, tMatrixD, tMatrixIp],
                                     [pMatrixI, pMatrixD, None]
                                     ])

    del rMatrixD, rMatrixI1, rMatrixI2
    del tMatrixD, tMatrixIr, tMatrixIp
    del pMatrixD, pMatrixI

    IdCores = []
    for i in range(3):
        IdCores.append(np.eye(polyOrder[i]))

    potentialY = poissonSolution(polyOrder, IdCores, elem,
                                  integrPoints=integrPoints, solutionTT_ranks=1, invRanks=1, rounding=False)
    for k in range(3):
        shape = potentialY[k].shape
        fig, axs = plt.subplots(shape[0], shape[-1])
        if( k == 0):
            fig.suptitle('r cores')
        for i in range(shape[0]):
            for j in range(shape[-1]):
                #print(i, j)
                if(shape[0] == 1 and shape[-1] != 1):
                    axs[j].imshow(potentialY[k][i, :, :, j].T)
                elif(shape[-1] == 1 and shape[0] != 1):
                    axs[i].imshow(potentialY[k][i, :, :, j].T)
                elif(shape[-1] != 1 and shape[0] != 1):
                    axs[i, j].imshow(potentialY[k][i, :, :, j].T)
                else:
                    plt .imshow(potentialY[k][i, :, :, j].T)
        # plt.colorbar()
        plt.show()

    poissonElem = element(np.array([[0, np.inf], [0, np.pi], [0, 2 * np.pi]]),
                   np.array(polyOrder, dtype=int), np.array([1, 0, 3]))
    print("poisson solution ranks")
    approx.printTT(potentialY)

    refGridR = elem[0].getMappedRefPoints()
    refGridToPoisGridUnit = poissonElem[0].inverseMap(refGridR)
    rShapeComponentOfY = potentialY[0].shape
    rComponentOfY = np.reshape(potentialY[0],
                    [rShapeComponentOfY[1], rShapeComponentOfY[2] * rShapeComponentOfY[3]])
    rComponentOfY = spec.barycentricChebInterpolate(rComponentOfY, x=refGridToPoisGridUnit, a=-1, b=1, axis=0)
    potentialY[0] = np.reshape(rComponentOfY,
                    [1, rShapeComponentOfY[1], rShapeComponentOfY[2], rShapeComponentOfY[3]])
    del rComponentOfY


    w, idNodes = integr.reg_32_wn(-1, 1, integrPoints)

    potentialY[0] = spec.barycentricChebInterpolateTensorAlongAxis(
        potentialY[0], idNodes, a=-1, b=1, axis=1)

    potentialY[1] = spec.barycentricChebInterpolateTensorAlongAxis(
        potentialY[1], idNodes, a=-1, b=1, axis=1)

    potentialY[2] = np.einsum('ij, ajkb -> aikb', elem[2].eval(idNodes), potentialY[2])
    #print("max and min of potential R core")

    print("potential shape")
    approx.printTT(potentialY)
    ttY = [None]*3
    for i in range(3):
        potentialY[i] = np.transpose(potentialY[i], [1, 0, 2, 3])
        if i < 2:
            ttY[i] = operations.integrateBilinearForm0_TensorWeight(
                elem, potentialY[i], integrPoints, axis=i, lambdaWeightAlongAxis=lambda x: x * x)
        else:
            ttY[i] = operations.integrateBilinearForm0_TensorWeight(
                elem, potentialY[i], integrPoints, axis=i)
        shape = ttY[i].shape

        # print(i, "s shape: ", shape)
        ttY[i] = np.reshape(ttY[i], [shape[0]*shape[1], shape[2], shape[3], shape[4]])
        shape = ttY[i].shape
        # print(i, "s shape after flattening: ", shape)

        ttY[i] = np.transpose(ttY[i], [1, 0, 2, 3])
        shape = ttY[i].shape
        # print(i, "s shape after axes rearrangement: ", shape)
    print("before separation")
    approx.printTT(ttY)

    approx.expandCoreMatrixForm(ttY, 0)
    approx.expandCoreMatrixForm(ttY, 2)
    approx.expandCoreMatrixForm(ttY, 4)
    ttY[0] = ttY[0][:, :-1, :-1, :]
    ttY[1] = ttY[1][:, :-1, :-1, :]
    print("after separation")
    approx.printTT(ttY)
    time.sleep(500)
    # ttY = approx.expandCoreMatrixForm(ttY, 0)

    # time.sleep(500)
    # tttY = []
    # for i in range(3):
    #     print(i, "s element")
    #     shape = ttY[i].shape
    #     print("preshape", ttY[i].shape)
    #     ttY[i] = np.reshape(ttY[i], [shape[0]*shape[1], shape[3]*shape[2]])
    #     #ttY[i] = np.transpose(ttY[i], [1, 2, 0, 3])
    #     aftershape = ttY[i].shape
    #     #ttY[i] = np.transpose(ttY[i], [0, 3, 1, 2, 4])
    #     print("aftershape", ttY[i].shape)
    #     flattenTTy = ttY[i].flatten()
    #
    #     # ttY[i] = np.reshape(ttY[i], [aftershape[0],
    #     #                              aftershape[1]*aftershape[2]*aftershape[3]])
    #     print("svd matrix shape")
    #     print(ttY[i].shape)
    #     #time.sleep(500)
    #     u, s, v = sp_lin.svd(ttY[i], full_matrices=False)
    #     #print(s[:30])
    #     cumsum = np.cumsum(s)
    #     r_delta = np.argmax(cumsum[-1] - cumsum < 1e-12) + 1
    #     u = np.dot(u[:, :r_delta], np.diag(np.sqrt(s[:r_delta])))
    #     v = np.dot(np.diag(np.sqrt(s[:r_delta])), v[:r_delta, :])
    #     print("separated core shapes")
    #     print(u.shape)
    #     print(v.shape)
    #     u = np.reshape(u, [shape[0], shape[1], u.shape[1]])
    #     u = np.transpose(u, (1, 0, 2))[np.newaxis, :, :, :]
    #     #u = np.reshape(u, [shape[0], shape[1], u.shape[1], u.shape[2]])
    #     #u = np.transpose(u, (2, 0, 1, 3))
    #
    #     v = np.reshape(v, [v.shape[0], shape[2], shape[3], shape[4]])
    #
    #     print(u.shape)
    #     print(v.shape)
    #     prod = np.einsum("ijkl, lmns -> ijkmns", u, v)
    #     #print(prod.shape)
    #     #print(np.max(prod.flatten() - flattenTTy))
    #     #if (i == 1):
    #         #time.sleep(500)
    #     if(i == 0):
    #         tttY.append(2*u[:, :-1, :-1, :])
    #         tttY.append(v[:, :-1, :-1, :])
    #     else:
    #         tttY.append(u)
    #         tttY.append(v)
    #     #print(u.shape)
    #     #print(v.shape)
    #     #time.sleep(500)
    # time.sleep(500)
    # approx.printTT(tttY)
    # ttY = tttY
    # del tttY
    #approx.printTT(ttY)
    ttFx = None
    core1 = None; core2 = None; core0 = None; grid0 = None; grid1 = None; grid2 = None

    Z = 2
    rMatrixI1 = operations.integrateBilinearForm0(elem, lambda x: -Z*2*x, integrPoints, 0)[:-1, :-1]
    rMatrixI2 = operations.integrateBilinearForm0(elem, lambda x: x * x, integrPoints, 0)[:-1, :-1]
    tMatrixI = operations.integrateBilinearForm0(elem, lambda x: np.sin(x) ** 2, integrPoints, 1)
    pMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 2)
    ttV1 = [rMatrixI1, rMatrixI2, tMatrixI, tMatrixI, pMatrixI, pMatrixI]
    ttV2 = [rMatrixI2, rMatrixI1, tMatrixI, tMatrixI, pMatrixI, pMatrixI]

    for i in range(len(ttV1)):
        ttV1[i] = ttV1[i][np.newaxis, :, :, np.newaxis]
        ttV2[i] = ttV2[i][np.newaxis, :, :, np.newaxis]

    rMatrixI1 = None; rMatrixI2 = None
    tMatrixI = None;  pMatrixI = None;

    rMatrixI = operations.integrateBilinearForm0(elem, lambda x: np.sqrt(2) * x * x, integrPoints, 0)[:-1, :-1]

    tMatrixI = operations.integrateBilinearForm0(elem, lambda x: np.sin(x) ** 2, integrPoints, 1)

    pMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 2)

    ttB = [rMatrixI, rMatrixI, tMatrixI, tMatrixI, pMatrixI, pMatrixI]
    for i in range(len(ttB)):
        ttB[i] = ttB[i][np.newaxis, : , :, np.newaxis]

    rMatrixI = None;    tMatrixI = None;    pMatrixI = None;

    #print("solving with TT als algorithm")
    t = time.time()

    ttSol = approx.eigAlterLeastSquares2d(A=ttA, B=ttB,
                                          ranks=2, sigma=-10, real=True, V=[ttY, ttV1, ttV2])
    print("energy: " , ttSol[0])

def solveEigenSphericalPoisForPlot(iStart, iEnd, iStep):
    for i in range(iStart, iEnd, iStep):
        #cores = solveEigenSphericalPois6d(np.array([8, 6, 4]), integrPoints=500)
        #for j in range(10):
        solveEigenSphericalPois6dRepulsionDifferentOrder(np.array([i, 8, 8]), integrPoints=500)
        #time.sleep(500)

solveEigenSphericalPoisForPlot(10, 22, 2)