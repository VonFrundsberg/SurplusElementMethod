import numpy as np
import scipy.interpolate as sp_interp
import scipy.fftpack as dft


def chebTransform(f, axis=0):
    n = f.shape[axis]
    a = 1 / (n - 1) * dft.dct(f[::-1], type=1, axis=axis)
    a[0] /= 2; a[-1] /= 2
    return a

def chebNodes(pointsAmount, a, b):
    nodes = (np.cos(np.arange(0, pointsAmount) * np.pi / (pointsAmount - 1))[::-1]) * (b - a) / 2
    nodes -= nodes[0]
    nodes += a
    return nodes

def periodicNodes(pointsAmount: int, halfInterval=False):
    if not halfInterval:
        return np.arange(pointsAmount)*2*np.pi/pointsAmount
    else:
        return np.arange(pointsAmount)*np.pi/pointsAmount
def barycentricChebInterpolate(f, x, a, b, extrapolation=0, axis=0):
    """

        Arguments:
            f:
            x:
            a:
            b:

        Returns:
            result: interpolated values of f at x
    """
    chebyshevPoints = chebNodes(pointsAmount=f.shape[0], a=a, b=b)
    extrapolatedPoints = np.argwhere((x < a) | (x > b))
    # print(extrapolatedPoints)
    result = sp_interp.barycentric_interpolate(chebyshevPoints, f, np.round(x, 32), axis=axis)
    match extrapolation:
        case 0:
            if len(result.shape) > 1:
                result[extrapolatedPoints, :] = 0
            elif len(result.shape) == 1:
                result[extrapolatedPoints] = 0
    return result

def barycentricChebInterpolateTensorAlongAxis(tensor, x, a, b, axis=0):
    """

        Arguments:
            f:
            x:
            a:
            b:

        Returns:
            result: interpolated values of f at x
    """
    chebyshevPoints = chebNodes(pointsAmount=tensor.shape[axis], a=a, b=b)
    extrapolatedPoints = np.argwhere((x < a) | (x > b))
    result = sp_interp.barycentric_interpolate(chebyshevPoints, tensor, x, axis=axis)
    #match extrapolation:
    #    case 0: result[extrapolatedPoints, :] = 0
    return result

def periodicInterpolate(f, x, extrapolation=0, axis=0):
    """Calculates partial derivative of element basis functions along axis.

        Arguments:
            f:
            x:
            a:
            b:

        Returns:
            result: interpolated values of f at x
    """
    chebyshevPoints = chebNodes(pointsAmount=f.shape[0], a=a, b=b)
    extrapolatedPoints = np.argwhere((x < a) | (x > b))
    result = sp_interp.barycentric_interpolate(chebyshevPoints, f, x, axis=axis)
    match extrapolation:
        case 0: result[extrapolatedPoints, :] = 0
    return result

def chebDiffMatrix(matrixSize, a=-1, b=1):
        n = matrixSize
        x = chebNodes(n, a, b)
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


def periodicDiffMatrix(matrixSize, halfInterval=False):

    n = matrixSize
    i = np.arange(n)
    j = np.arange(n)
    if not halfInterval:
        x = periodicNodes(n)
        diffMatrix = 0.5*(-1)**(i[:, np.newaxis] + j[np.newaxis, :])/ \
                     (np.tan((x[:, np.newaxis] - x[np.newaxis, :])/2.0))
        np.fill_diagonal(diffMatrix, 0)
    else:
        x = periodicNodes(n, True)
        diffMatrix = (-1) ** (i[:, np.newaxis] + j[np.newaxis, :]) / \
                     (np.tan((x[:, np.newaxis] - x[np.newaxis, :])))
        np.fill_diagonal(diffMatrix, 0)
    return diffMatrix
def periodic2DiffMatrix(matrixSize, halfInterval=False):
    n = matrixSize
    i = np.arange(n)
    j = np.arange(n)
    if not halfInterval:
        x = periodicNodes(n)
        diff2Matrix = -(-1)**(i[:, np.newaxis] + j[np.newaxis, :])* \
                     0.5*(np.power(np.sin((x[:, np.newaxis] - x[np.newaxis, :])/2.0), -2))
        diag = (-n**2 - 2)/12.0
        np.fill_diagonal(diff2Matrix, diag)
    else:
        x = periodicNodes(n, halfInterval=True)
        diff2Matrix = -(-1) ** (i[:, np.newaxis] + j[np.newaxis, :]) * \
                      2 * (np.power(np.sin((x[:, np.newaxis] - x[np.newaxis, :])), -2))
        diag = (-n ** 2 - 2) / 3.0
        np.fill_diagonal(diff2Matrix, diag)
    return diff2Matrix