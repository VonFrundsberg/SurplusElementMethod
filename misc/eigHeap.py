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
from heap import solveSphericalPoisVecRHS as poissonSolution
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
    print("init energy: " , ttSol[0] + 4)
    solCores = []
    for it in ttSol[1].cores:
        shape = it.shape
        solCores.append(np.reshape(it, [shape[0], shape[1], shape[3]]))
    shape1 = solCores[0].shape
    shape2 = solCores[3].shape
    solCores[0] = np.hstack([solCores[0], np.zeros(shape=(shape1[0], 1, shape1[2]))])
    solCores[3] = np.hstack([solCores[3], np.zeros(shape=(shape2[0], 1, shape2[2]))])
    return solCores


def solveEigenSphericalPois6dRepulsion(polyOrder, prevSolutionCores, integrPoints=350):

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

    potentialY = poissonSolution(polyOrder, prevSolutionCores[:3], elem,
                                 integrPoints=integrPoints, solutionTT_ranks=6)
    for it in potentialY:
        print(it.shape)
    time.sleep(500)
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
def solveEigenSphericalPoisForPlot(iStart, iEnd, iStep):
    for i in range(iStart, iEnd, iStep):
        cores = solveEigenSphericalPois6d(np.array([i, 4, 4]), integrPoints=500)
        for j in range(5):
            cores = solveEigenSphericalPois6dRepulsion(np.array([i, 4, 4]), cores, integrPoints=500)
        time.sleep(500)

solveEigenSphericalPoisForPlot(4, 30, 2)