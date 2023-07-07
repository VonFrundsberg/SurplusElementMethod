import numpy as np
import scipy.interpolate as sp_interp
import scipy.fftpack as dft


def chebTransform(f, axis=0):
    n = f.shape[axis]
    a = 1 / (n - 1) * dft.dct(f[::-1], type=1, axis=axis)
    a[0] /= 2; a[-1] /= 2
    return a

def calcChebPoints(pointsAmount, a, b):
    nodes = (np.cos(np.arange(0, pointsAmount) * np.pi / (pointsAmount - 1))[::-1]) * (b - a) / 2
    nodes -= nodes[0]
    nodes += a
    return nodes

def barycentricChebInterpolate(f, x, a, b, extrapolation=0):
    """Calculates partial derivative of element basis functions along axis.

        Arguments:
            f: values of function at chebyshev points, algebraically rescaled to an [a, b] interval
            x: points at which interpolated function is evaluated
            a, b: range of chebyshev points

        Returns:
            result: interpolated values of f at x
    """
    chebyshevPoints = calcChebPoints(pointsAmount=f.size, a=a, b=b)
    extrapolatedPoints = np.argwhere((x < a) | (x > b))
    result = sp_interp.barycentric_interpolate(chebyshevPoints, f, x, axis=0)
    match extrapolation:
        case 0: result[extrapolatedPoints] = 0
    return result

def ChebDiffMatrix(matrixSize, a=-1, b=1):
        n = matrixSize
        x = calcChebPoints(n, a, b)
        X = np.ones([n, n], dtype=float)
        X = ((X.T)*x).T
        dX = X - X.T
        C = np.append([2], np.ones(n - 2))
        C = np.append(C, [2])
        C *= (-1)**np.arange(0, n)
        C = np.reshape(np.kron(C, 1/C), newshape=[n, n])
        D = C/(dX + np.eye(n))
        D = D - np.diag(np.sum(D, axis=1))
        return D
