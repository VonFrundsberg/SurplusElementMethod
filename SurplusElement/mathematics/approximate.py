import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.linalg as sp_linalg
import time as time
import scikit_tt.solvers.sle as sle
import scikit_tt.solvers.evp as evp
from scikit_tt.tensor_train import TT
import scikit_tt.tensor_train as tt


def printTT(argTensor):
    for it in argTensor:
        print(it.shape)
def simpleTTsvd(argTensor, tol=1e-6, R_MAX=100, makeCopy=True):
    tensor = argTensor.copy()
    shape = tensor.shape
    dim = len(shape)
    r = np.ones(dim, dtype=int)
    leftShapeProd = np.prod(shape)
    cores = []
    for i in range(dim - 1, 0, -1):
        tensor = np.reshape(tensor, [int(leftShapeProd/r[i]/shape[i]), shape[i]*r[i]])
        u, s, v = sp_linalg.svd(tensor, full_matrices=False)
        cumsum = np.cumsum(s)
        r_delta = np.argmax(cumsum[-1] - cumsum < tol) + 1
        r[i - 1] = min(R_MAX, r_delta)
        if r[i-1] == R_MAX:
            print(i, " rank exceeded R_MAX, TT approximation error may be high")

        cores.append((np.reshape(np.diag(np.sqrt(s[:r[i - 1]])) @ v[: r[i - 1]], [r[i-1], shape[i], r[i]])))
        leftShapeProd = leftShapeProd * r[i - 1]/shape[i]/r[i]
        tensor = np.dot(u[:, : r[i - 1]], np.diag(np.sqrt(s[:r[i-1]])))
    cores.append(np.reshape(tensor, [1, shape[0], r[0]]))
    cores = cores[::-1]
    return cores

def matrixToVectorTT(argMatrixTensor):
    """Only for cases where inner indices are the same
            """
    cores = []
    for i in range(len(argMatrixTensor)):
        shape = argMatrixTensor[i].shape
        core = np.transpose(argMatrixTensor[i], (1, 2, 0, 3))
        core = np.reshape(core, [int(shape[1]**2), shape[0], shape[3]])
        cores.append(np.transpose(core, (1, 0, 2)))
    return cores

def vectorToMatrixTT(argVectorTensor):
    """Only for cases where inner indices are expected to be the same
                """
    cores = []
    for i in range(len(argVectorTensor)):
        shape = argVectorTensor[i].shape
        core = np.transpose(argVectorTensor[i], (1, 0, 2))
        sqrt = int(np.sqrt(shape[1]))
        core = np.reshape(core, [sqrt, sqrt, shape[0], shape[2]])
        cores.append(np.transpose(core, (2, 0, 1, 3)))
    return cores
def hadamardProduct(lhsTTarg, rhsTTarg, matrixForm = False):
    cores = []

    if matrixForm == True:
        lhsTT = matrixToVectorTT(lhsTTarg)
        rhsTT = matrixToVectorTT(rhsTTarg)
    for i in range(len(lhsTT)):
        lhsShape = lhsTT[i].shape
        rhsShape = rhsTT[i].shape
        #core = np.einsum('ijkl, mkn -> jimln', A[i], f[i])
        result = np.einsum("ijk, mjn -> jimkn", lhsTT[i], rhsTT[i])
        resultShape = result.shape
        result = np.reshape(result, [resultShape[0],
                                     resultShape[1]*resultShape[2],
                                     resultShape[3]*resultShape[4]])
        cores.append(np.transpose(result, [1, 0, 2]))
    if matrixForm == True:
        cores = vectorToMatrixTT(cores)
    return cores

def simpleQTTsvd(argTensor, tol=1e-6, makeCopy=True, flattenNewaxes=None):
    """Calculates QTT approximation to argTensor, elements of shape of argTensor should be the same
    """
    shape = argTensor.shape
    dim = len(shape)
    log2arr = np.log2(shape)
    tensor = argTensor.copy()
    if all(np.equal(np.mod(log2arr, 1), 0)):
        pass
    else:
        print("some dimension is not a power of 2")
        return ValueError
    newshape = np.array(2 * np.ones(int(np.sum(log2arr))), dtype=int)
    tensor = np.reshape(tensor, newshape)
    #######if we need reordering of axes, else keep commented
    # newaxes = np.arange(np.sum(log2arr))
    # flattenNewaxes = np.ones(newaxes.size, dtype=int)
    # newaxes = np.reshape(newaxes, [len(log2arr), int(log2arr[0])])
    # for i in range(newaxes.shape[1]):
    #     flattenNewaxes[(i)*len(log2arr): (i + 1)*len(log2arr)] = np.array(newaxes[:, i])
    # tensor = np.transpose(tensor, flattenNewaxes)
    shape = tensor.shape
    dim = len(shape)
    r = np.ones(dim, dtype=int)
    leftShapeProd = np.prod(shape)
    cores = []
    R_MAX = 100
    for i in range(dim - 1, 0, -1):
        tensor = np.reshape(tensor, [int(leftShapeProd / r[i] / shape[i]), shape[i] * r[i]])
        u, s, v = sp_linalg.svd(tensor, full_matrices=False)

        cumsum = np.cumsum(s ** 2)
        r_delta = np.argmax(cumsum[-1] - cumsum < tol**2) + 1
        r[i - 1] = min(R_MAX, r_delta)

        cores.append(np.reshape(v[: r[i - 1]], [r[i - 1], shape[i], r[i]]))
        leftShapeProd = leftShapeProd * r[i - 1] / shape[i] / r[i]
        tensor = np.dot(u[:, : r[i - 1]], np.diag(s[:r[i - 1]]))
    cores.append(np.reshape(tensor, [1, shape[0], r[0]]))
    cores = cores[::-1]

    return cores
def vectorTTsvd(f, tol=1e-6):
    shape = f.shape
    ls = len(shape)
    f = f.copy()
    sigma = 1
    cores = []
    for i in range(len(shape) - 1):

        f = np.reshape(f, [np.prod(shape[:-i - 1]), shape[-1 - i]*sigma])

        u, s, v = sp_linalg.svd(f, full_matrices=False)

        psigma = sigma
        sigma = max(1, np.size(s[np.abs(s) > tol]))
        f = np.dot(u[:, :sigma], np.diag(np.sqrt(s[:sigma])))
        v = np.dot(np.diag(np.sqrt(s[:sigma])), v[:sigma, :])

        v = np.reshape(v, [sigma, shape[ls - i - 1], psigma])

        cores.append(v)

        f = np.reshape(f, [*shape[:-1 - i], sigma])

    f = np.reshape(f, [1, *shape[:-1 - i], sigma])
    cores.append(f)
    return cores[::-1]
    # for i in range(len(cores)):
    #     print(cores[i].shape)
def matrixTTsvd(A, shape, tol=1e-6, f=None):
    A = np.reshape(A, [*shape, *shape])
    first = np.arange(shape.size)
    last = np.arange(shape.size, 2*shape.size)
    newAxes = np.zeros(2*shape.size)
    newAxes[::2] = first; newAxes[1::2] = last
    newAxes = np.array(newAxes, dtype=int)
    A = np.transpose(A, newAxes)
    A = np.reshape(A, shape**2)
    A = simpleTTsvd(A, tol)
    for i in range(len(A)):
        a = A[i].shape[0]; b = A[i].shape[2]
        A[i] = np.reshape(A[i], [a, shape[i], shape[i], b])
    return A
def meshgrid(*arg):
    return np.meshgrid(*arg, indexing='ij')

def integrateTT(integratedTTarg, weights, matrixForm=False):
    """not implemented"""
    if matrixForm == True:
        integratedTT = matrixToVectorTT(integratedTTarg)
    else:
        integratedTT = integratedTTarg
    integralsList = []
    for i in range(len(integratedTT)):
         integralsList.append(np.einsum("ijk, j -> ik", integratedTT[i], weights[i]))
    printTT(integralsList)
    #result = integralsList[0]
    #for i in range(1, len(integralsList)):
    #    result = result.dot(integralsList[i])
    #return result
def contraction(u, a):
    v = []
    for i in range(len(u)):
        tmp1 = (a[i].T)[:, None, :, None]
        tmp2 = u[i][None]
        v.append(np.sum(tmp1*tmp2, axis=2))
    res = v[0]
    for i in range(1, len(u)):
        res = np.dot(res, v[i])
    return np.squeeze(np.array(res))

def kronSumtoTT(A: list, B: list):
    """Construct TT cores of matrix sum
        [T_1, T_2, T_3] = B_1*A_2*A_3 + A_1*B_2*A_3 + A_1*A_2*B_3,
        where * is a kronecker product.
        Calculates for arbitrary list len (in example len(A) = 3). It must be that len(A) == len(B)
        Arguments:
            A:
            B:
        Returns:
            cores:
    """
    dim = len(A)
    if dim != len(B):
        print("List length is not the same")
        return None
    cores = []
    core = np.stack((B[0], A[0]), axis=2)[np.newaxis, :, :, :]
    cores.append(core)
    for i in range(1, dim - 1):
        core = np.stack((A[i], 0*B[i], B[i], A[i]), axis=2)[np.newaxis, :, :, :]
        core = np.reshape(core, [core.shape[1], core.shape[2], 2, 2])
        core = np.transpose(core, [2, 0, 1, 3])
        cores.append(core)
    core = np.stack((A[-1], B[-1]), axis=0)[:, :, :, np.newaxis]
    cores.append(core)
    return cores


def sphericalLaplace6d(matricesMatrix):
    """Construct TT cores of matrix sum for sum of two laplacians in spherical coordinates
        Returns:
            cores: with shapes G_1(r_1) G_2(r_2) G_3(theta_1) G_4(theta_2) G_5(phi_1) G_6(phi)_2
    """
    dim = 6
    M = matricesMatrix
    cores = []
    """R CORES
        """
    lhsCore = np.stack([M[0][1], M[0][2]])
    cores.append(np.transpose(lhsCore, [1, 2, 0])[np.newaxis, :, :, :])

    rhsCore = np.stack([M[0][2], M[0][1]])[:, :, :, np.newaxis]
    cores.append(rhsCore)

    cores.append(M[1][0][np.newaxis, :, :, np.newaxis])
    cores.append(M[1][0][np.newaxis, :, :, np.newaxis])

    cores.append(M[2][0][np.newaxis, :, :, np.newaxis])
    cores.append(M[2][0][np.newaxis, :, :, np.newaxis])
    ttR = TT(cores)
    """THETA CORES
            """
    cores = []
    lhsCore = np.stack([M[0][0], M[0][2]])
    cores.append(np.transpose(lhsCore, [1, 2, 0])[np.newaxis, :, :, :])

    zeros = np.zeros(M[0][0].shape)
    rhsCore = np.stack([M[0][2], zeros, zeros, M[0][0]])[:, :, :]
    rhsCore = np.reshape(rhsCore, [2, 2, *zeros.shape])
    rhsCore = np.transpose(rhsCore, [0, 2, 3, 1])

    cores.append(rhsCore)
    zeros = np.zeros(M[1][0].shape)
    thetaLhsCore = np.stack([M[1][1], zeros, zeros, M[1][0]])[:, :, :]
    thetaLhsCore = np.reshape(thetaLhsCore, [2, 2, *zeros.shape])
    thetaLhsCore = np.transpose(thetaLhsCore, [0, 2, 3, 1])
    cores.append(thetaLhsCore)
    thetaRhsCore = np.stack([M[1][0], M[1][1]])[:, :, :, np.newaxis]
    cores.append(thetaRhsCore)

    cores.append(M[2][0][np.newaxis, :, :, np.newaxis])
    cores.append(M[2][0][np.newaxis, :, :, np.newaxis])

    ttTheta = TT(cores)
    """PHI CORES
            """
    cores = []
    lhsCore = np.stack([M[0][0], M[0][2]])
    cores.append(np.transpose(lhsCore, [1, 2, 0])[np.newaxis, :, :, :])

    zeros = np.zeros(M[0][0].shape)
    rhsCore = np.stack([M[0][2], zeros, zeros, M[0][0]])[:, :, :]
    rhsCore = np.reshape(rhsCore, [2, 2, *zeros.shape])
    rhsCore = np.transpose(rhsCore, [0, 2, 3, 1])

    cores.append(rhsCore)
    zeros = np.zeros(M[1][0].shape)
    thetaLhsCore = np.stack([M[1][2], zeros, zeros, M[1][0]])[:, :, :]
    thetaLhsCore = np.reshape(thetaLhsCore, [2, 2, *zeros.shape])
    thetaLhsCore = np.transpose(thetaLhsCore, [0, 2, 3, 1])
    cores.append(thetaLhsCore)

    thetaRhsCore = np.stack([M[1][0], zeros, zeros, M[1][2]])[:, :, :]
    thetaRhsCore = np.reshape(thetaRhsCore, [2, 2, *zeros.shape])
    thetaRhsCore = np.transpose(thetaRhsCore, [0, 2, 3, 1])
    cores.append(thetaRhsCore)

    zeros = np.zeros(M[2][0].shape)
    phiLhsCore = np.stack([M[2][1], zeros, zeros, M[2][0]])[:, :, :]
    phiLhsCore = np.reshape(phiLhsCore, [2, 2, *zeros.shape])
    phiLhsCore = np.transpose(phiLhsCore, [0, 2, 3, 1])
    cores.append(phiLhsCore)
    phiRhsCore = np.stack([M[2][0], M[2][1]])[:, :, :, np.newaxis]
    cores.append(phiRhsCore)

    ttPhi = TT(cores)
    return ttR + ttTheta + ttPhi
def kronSumtoTT_blockFormat(matricesMatrix):
    """Construct TT cores of matrix sum. May work only in 3d case

        Returns:
            cores:
    """
    dim = len(matricesMatrix)
    M = matricesMatrix
    cores = []
    core = np.stack((M[0][1], M[0][2]), axis=2)[np.newaxis, :, :, :]
    cores.append(core)
    for i in range(1, dim - 1):
        core = np.stack((M[i][0], 0*M[i][1], M[i][1], M[i][2]), axis=2)[np.newaxis, :, :, :]
        core = np.reshape(core, [core.shape[1], core.shape[2], 2, 2])
        core = np.transpose(core, [2, 0, 1, 3])
        cores.append(core)
    core = np.stack((M[2][0], M[2][1]), axis=0)[:, :, :, np.newaxis]
    cores.append(core)
    return cores

def invertTT_Matrix(A, rankOfInverse, rounding=False):
    """"calculates TT approximation to matrix inverse by vectorizing the matrix equation

    Oseledets, Ivan V., and Sergey V. Dolgov.
     "Solution of linear systems and matrix inversion in the TT-format."
     SIAM Journal on Scientific Computing 34.5 (2012): A2718-A2739.
     """
    ttA = TT(A)
    rhsTT = tt.eye(ttA.col_dims)
    ttA = None
    solMatrix = []
    for i in range(len(A)):
        shape = A[i].shape
        coreOfMatrix = np.einsum("ijkl, op -> jokpil",A[i], np.eye(A[i].shape[1]))
        coreOfMatrix = np.reshape(coreOfMatrix, [shape[1]**2, shape[2]**2, shape[0], shape[3]])
        solMatrix.append(np.transpose(coreOfMatrix, [2, 0, 1, 3]))
    coreOfMatrix = None
    for i in range(len(rhsTT.cores)):
        shape = (rhsTT.cores[i]).shape
        rhsTT.cores[i] = np.reshape(rhsTT.cores[i], [1, shape[1]**2, 1, 1])
    rhsTT = TT(rhsTT.cores)
    ttSolMatrix = TT(solMatrix)
    solMatrix = None
    initTT = tt.ones(ttSolMatrix.row_dims, [1] * ttSolMatrix.order, ranks=rankOfInverse).ortho_right()
    solution = sle.als(operator=ttSolMatrix, initial_guess=initTT, right_hand_side=rhsTT)

    sol = []
    for i in range(len(solution.cores)):
        shape = (solution.cores[i]).shape
        solCore = np.transpose(solution.cores[i], [1, 2, 0, 3])
        sqrt = int(np.sqrt(shape[1]))
        solCore = np.reshape(solCore, [sqrt, sqrt, shape[0], shape[3]])
        solCore = np.transpose(solCore, [2, 0, 1, 3])
        sol.append(solCore)
    if rounding == True:
        vecRound(sol, tol=1e-6, matrixForm=True)
    return sol
def toFullTensor(u, matrixForm=False):
    T = u[0]
    for k in range(len(u) - 1):
        T = np.tensordot(T, u[k + 1], axes=1)
    T = np.squeeze(T)
    if matrixForm == True:
        shape = T.shape
        arange = np.reshape(np.arange(len(shape)), [int(len(shape)/2), 2])
        T = np.transpose(T, np.concatenate((arange[:, 0], arange[:, 0] + 1), axis=0))
    return T

def expandCoreMatrixForm(argTensor, expandIndex: int):
    """
        expands core given by index into two dims where
        ...(alpha, n_1, n_2, beta)... is init form of core
        the result will be in the form
       ... (alpha, n_1_i, n_1_j, gamma) X (gamma, n_2_i, n_2_j, gamma)...
        """
    alpha = argTensor[expandIndex].shape[0]; beta = argTensor[expandIndex].shape[-1];


    cores = argTensor
    # print("init cores")
    ttCores = TT(cores)
    #printTT(ttCores.cores)
    # fullTensor = ttCores.full()
    # print("initial kronTensor shape", fullTensor.shape)

    #reshapedLhs = np.transpose(lhs, (0, 1, 3, 2))
    #print("prev shape", lhs.shape)
    """
    separation of core
    """
    expandShape = argTensor[expandIndex].shape
    reshapedCore = np.reshape(argTensor[expandIndex],
                              [alpha*expandShape[1], beta*expandShape[2]])
    u, s, v = sp_linalg.svd(reshapedCore, full_matrices=False)
    # print(s[:30])
    cumsum = np.cumsum(s)
    r_delta = np.argmax(cumsum[-1] - cumsum < 1e-12) + 1
    #r_delta = 1000
    u = np.dot(u[:, :r_delta], np.diag(np.sqrt(s[:r_delta])))
    v = np.dot(np.diag(np.sqrt(s[:r_delta])), v[:r_delta, :])
    # print("separated shapes at left core")
    # print(u.shape, v.shape)
    # time.sleep(500)
    sqrtU = int(np.sqrt(expandShape[1]))
    u = np.reshape(u, [alpha, sqrtU, sqrtU, u.shape[1]])
    #u = np.transpose(u, (0, 2, 1, 3))
    sqrtV = int(np.sqrt(expandShape[2]))
    v = np.reshape(v, [v.shape[0], sqrtV, sqrtV, beta])


    # print("resulting separated core shapes")
    #printTT([argTensor[0], u, v, argTensor[2]])
    argTensor[expandIndex] = v
    argTensor.insert(expandIndex, u)
    #newTensor = TT([argTensor[0], u, v, argTensor[2]])
    # print("cores of newTensor")
    # printTT(newTensor.cores)
    #return newTensor
    # print("newTensor shape")
    # print(newTensor.full().shape)

    #print(np.max(np.abs(newTensor.full().flatten() - fullTensor.flatten())))
    # plt.scatter(np.arange(fullTensor.size),
    #            newTensor.full().flatten() - fullTensor.flatten())
    # plt.show()


    pass
def vecRound(u, tol=1e-6, matrixForm=False):
    if matrixForm == True:
        for i in range(len(u)):
            shape = u[i].shape
            u[i] = np.transpose(u[i], [1, 2, 0, 3])
            u[i] = np.reshape(u[i], [shape[1]**2, shape[0], shape[3]])
            u[i] = np.transpose(u[i], [1, 0, 2])

    size = len(u)
    for i in range(size - 1, 1, -1):
        a, n, b = u[i].shape
        # R, Q = sp_linalg.rq(np.reshape(u[i], [a, n*b]))
        R, Q = sp_linalg.qr(np.reshape(u[i], [a, n * b]), mode='economic')
        u[i] = np.reshape(Q, [int(Q.shape[0]), n, 1])
        # print('r', R.shape)
        # print(u[i - 1].shape, Q.shape, R.shape)
        u[i - 1] = np.tensordot(u[i - 1], R, axes=(2, 0))
        # print(Q.shape, R.shape)
    for i in range(size - 1):
        a, n, b = u[i].shape
        # print(a, n, b)
        U, S, V = sp_linalg.svd(np.reshape(u[i], [a*n, b]), full_matrices=False)
        # print(i, S)
        sigma = min(max(1, np.size(S[np.abs(S) > tol])) + 1, np.size(S))
        #sigma = max(1, np.size(s[np.abs(s) > tol]))
        u[i] = U[:, :sigma]; V = np.dot(np.diag(S[:sigma]), V[:sigma, :])
        # print(V.shape, u[i + 1].shape)
        u[i + 1] = np.tensordot(V, u[i + 1], axes=(1, 0))
        u[i] = np.reshape(u[i], [a, n, sigma])
    if(matrixForm == True):
        for i in range(len(u)):
            shape = u[i].shape
            u[i] = np.transpose(u[i], [1, 0, 2])
            u[i] = np.reshape(u[i], [int(np.sqrt(shape[1])),
                                     int(np.sqrt(shape[1])), shape[0], shape[2]])
            u[i] = np.transpose(u[i], [2, 0, 1, 3])
def alterLeastSquares(A, f, ranks):
    ttA = TT(A)
    newf = []
    for i in range(len(f)):
        newf.append(f[i][:, :, np.newaxis, :])
    ttF = TT(newf)
    initTT = tt.ones(ttA.row_dims, [1] * ttA.order, ranks=ranks).ortho_right()
    sol = sle.als(ttA, initTT, ttF)
    return sol
def eigAlterLeastSquares(A, B, ranks, sigma=1, V = None, prev = None, real=True, shift=None):
    ttA = TT(A)
    if V is not None:
        ttA = ttA + TT(V)
    ttB = TT(B)
    initTT = tt.ones(ttA.row_dims, [1] * ttA.order, ranks=ranks).ortho_right()
    if prev is not None:
        res = evp.als(operator=ttA, initial_guess=initTT,
                      operator_gevp=ttB, sigma=sigma, real=real,
                      previous=prev, shift=100, repeats=30,
                      conv_eps=1e-10, number_ev=20)
    else:
        res = evp.als(operator=ttA, initial_guess=initTT, operator_gevp=ttB,
                      sigma=sigma, shift=1, repeats=20)
        # res = evp.als(operator=ttA, initial_guess=initTT,
        #               operator_gevp=ttB, sigma=sigma, real=real,
        #               shift=1, repeats=20,
        #               conv_eps=1e-10, number_ev=20)
    return res


def eigAlterLeastSquares2d(A, B, ranks, sigma=1, V = None, prev = None, real=True, shift=None):
    #ttA = A
    if len(A) == 2:
        ttA = TT(A[0]) + TT(A[1])
    else:
        ttA = TT(A)
    #ttA = TT(A)
    A = None
    if V is not None:
        for i in range(len(V)):
            ttA = ttA + TT(V[i])

    V = None

    if(len(B) == 2):
        ttB = TT(B[0])
        for i in range(1, len(B)):
                ttB = ttB + TT(B[i])
    else:
        ttB = TT(B)
    B = None
    initTT = tt.ones(ttA.row_dims, [1] * ttA.order, ranks=ranks).ortho_right()
    if prev is not None:
        res = evp.als(operator=ttA, initial_guess=initTT,
                      operator_gevp=ttB, sigma=sigma, real=real, previous=prev, shift=shift)
    else:
        res = evp.als(operator=ttA, initial_guess=initTT,
                      operator_gevp=ttB, sigma=sigma, repeats=20)
    return res