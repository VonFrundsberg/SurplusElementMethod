import numpy as np
from pathlib import Path
#main reads data from integrationData.txt to data = {}
import importlib.resources
data = {}

def __main():
    # Open and read the file from the package
    with importlib.resources.path('SurplusElement.mathematics', 'integrationData.txt') as file_path:
        with open(file_path, 'r') as file:
            while True:
                spec = file.readline()
                if spec == "":
                    break
                spec = spec.split()
                shape = int(spec[1])

                # Read arrays from file using numpy
                x = np.fromfile(file, count=shape, sep=" \n")
                w = np.fromfile(file, count=shape, sep=" \n")
                data[spec[0]] = x, w

            # Return or process 'data' as needed
            # return data

__main()
#Gauss-Trapezoidal rule f: K->K
#h^{32}
def reg_32(f, a=-1, b=1, n=20):
    x, w = data["reg_32"]
    n = n
    A = 14
    h = 1 / (n + 2 * A - 1)
    points = np.append((b - a) * x * h + a, (b - a) * (A * h + np.arange(n) * h) + a)
    points = np.append(points, ((b - a) * (1 - x * h) + a)[::-1])
    S = f(points)
    if len(S.shape) > 1:
        return (b - a) * h * (
            np.dot(S[:, 0: 16], w) + np.dot(S[:, 16 + n:32 + n], w[::-1]) + np.sum(S[:, 16: n + 16], axis=1))
    else:
        return (b - a) * h * (np.dot(S[0: 16], w) + np.dot(S[16 + n:32 + n], w[::-1]) + np.sum(S[16: n + 16]))

#Gauss-Trapezoidal rule. Returns weights and nodes
#h^{32}
def reg_32_wn(a=-1, b=1, n=20):
    x, w = data["reg_32"]
    A = 14
    h = 1 / (n + 2 * A - 1)

    left_points = (b - a) * x * h + a
    mid_points = (b - a) * (A * h + np.arange(n) * h) + a
    right_points = ((b - a) * (1 - x * h) + a)[::-1]

    left_weights = w * (b - a) * h
    mid_weights = np.ones(n) * (b - a) * h
    right_weights = w[::-1] * (b - a) * h
    points = np.hstack([left_points, mid_points, right_points])
    weights = np.hstack([left_weights, mid_weights, right_weights])
    # points = np.concatenate(points).ravel()
    # weights = np.concatenate(weights).ravel()
    return weights, points


def log_16_wn(a=-1, b=1, n=20):
    xLog, wLog = data["log_16"]
    xReg, wReg = data["reg_32"]
    A = 10
    B = 14
    h = 1 / (n + A + B - 1)
    left_points = (b - a) * xLog * h + a
    mid_points = (b - a) * (A * h + np.arange(n) * h) + a

    right_points = ((b - a) * (1 - xReg * h) + a)[::-1]

    left_weights = wLog * (b - a) * h
    mid_weights = np.ones(n) * (b - a) * h
    right_weights = wReg[::-1] * (b - a) * h
    points = np.hstack([left_points, mid_points, right_points])
    weights = np.hstack([left_weights, mid_weights, right_weights])
    return weights, points

def reg_22_wn(a=-1, b=1, n=20):
    x, w = data["reg_22"]
    A = 21
    h = 1 / (n + 2 * A - 1)
    left_points = (b - a) * x * h + a
    mid_points = (b - a) * (A * h + np.arange(n) * h) + a
    right_points = ((b - a) * (1 - x * h) + a)[::-1]

    left_weights = w * (b - a) * h
    mid_weights = np.ones(n) * (b - a) * h
    right_weights = w[::-1] * (b - a) * h
    points = np.hstack([left_points, mid_points, right_points])
    weights = np.hstack([left_weights, mid_weights, right_weights])
    return weights, points

def clenshaw_wn(n=20):
    nn = np.arange(0, n)
    w = -(1 + (-1)**nn)/(nn**2 - 1)
    w = np.nan_to_num(w)
    x = (np.cos(np.arange(0, n) * np.pi / (n - 1))[::-1])
    return w, x


