import numpy as np
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
        cores.append(np.reshape(v[: r[i - 1]], [r[i-1], shape[i], r[i]]))
        leftShapeProd = leftShapeProd * r[i - 1]/shape[i]/r[i]
        tensor = np.dot(u[:, : r[i - 1]], np.diag(s[:r[i-1]]))
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
def kronSumtoTT_blockFormat(matricesMatrix):
    """Construct TT cores of matrix sum

        Returns:
            cores:
    """
    dim = len(matricesMatrix)
    M = matricesMatrix
    cores = []
    core = np.stack((M[0][1], M[0][2]), axis=2)[np.newaxis, :, :, :]
    cores.append(core)
    for i in range(1, dim - 1):
        core = np.stack((M[i][0], 1/2*M[i][1], 1/2*M[i][1], M[i][2]), axis=2)[np.newaxis, :, :, :]
        core = np.reshape(core, [core.shape[1], core.shape[2], 2, 2])
        core = np.transpose(core, [2, 0, 1, 3])
        cores.append(core)
    core = np.stack((M[2][0], M[2][1]), axis=0)[:, :, :, np.newaxis]
    cores.append(core)
    return cores

def invertTT_Matrix(A, rankOfInverse):
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
                      operator_gevp=ttB, sigma=sigma, real=real, previous=prev, shift=shift)
    else:
        res = evp.als(operator=ttA, initial_guess=initTT, operator_gevp=ttB, sigma=sigma, real=real)
    return res


def eigAlterLeastSquares2d(A, B, ranks, sigma=1, V = None, prev = None, real=True, shift=None):
    ttA = TT(A[0]) + TT(A[1])
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
        res = evp.als(operator=ttA, initial_guess=initTT, operator_gevp=ttB, sigma=sigma, repeats=10)
    return res