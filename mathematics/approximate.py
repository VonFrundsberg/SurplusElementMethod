import numpy as np
import scipy.linalg as sp_linalg
import time as time

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

def toFullTensor(u):
    T = u[0]
    for k in range(len(u) - 1):
        T = np.tensordot(T, u[k + 1], axes=1)
    T = np.squeeze(T)
    return T