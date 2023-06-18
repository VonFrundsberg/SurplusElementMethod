import numpy as np
import scipy.interpolate as sp_interp
import scipy.fftpack as dft


def chebTransform(f, axis=0):
    n = f.shape[axis]
    a = 1 / (n - 1) * dft.dct(f[::-1], type=1, axis=axis)
    a[0] /= 2; a[-1] /= 2
    return a

def chebNodes(n, a=-1, b=1):

    nodes = (np.cos(np.arange(0, n) * np.pi / (n - 1))[::-1]) * (b - a) / 2
    nodes -= nodes[0]
    nodes += a
    return nodes

def bary(f, x, a=-1, b=1, cx=None):
    if cx is None:
        if len(f.shape) > 1:
            cx = chebNodes(n=f.shape[0], a=a, b=b)
        else:
            cx = chebNodes(n=f.size, a=a, b=b)
    args0 = np.argwhere((x < a) | (x > b))
    res = sp_interp.barycentric_interpolate(cx, f, x, axis=0)
    res[args0] = 0
    return res

def chebDiff(n, a=-1, b=1):
        x = chebNodes(n, a, b)
        X = np.ones([n, n], dtype=np.float)
        X = ((X.T)*x).T
        dX = X - X.T
        C = np.append([2], np.ones(n - 2))
        C = np.append(C, [2])
        C *= (-1)**np.arange(0, n)
        C = np.reshape(np.kron(C, 1/C), newshape=[n, n])
        D = C/(dX + np.eye(n))
        D = D - np.diag(np.sum(D, axis=1))
        return D
