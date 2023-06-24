import numpy as np
import scipy.linalg as sp_linalg

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
    newAxes = np.array(newAxes, dtype=np.int)
    A = np.transpose(A, newAxes)
    A = np.reshape(A, shape**2)
    A = vectorTTsvd(A, tol)
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

# n = 100
# x = np.linspace(0, 1, n)
# y = np.linspace(0, 1, n)
# z = np.linspace(0, 1, n)
# xx, yy, zz = np.meshgrid(x, y, z)
# f = np.sqrt(xx**2 + yy**2 + zz**2)
# ttF = vectorTTsvd(f, tol=1e-3)
# for it in ttF:
#     print(it.shape)