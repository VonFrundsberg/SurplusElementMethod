import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
import scipy.linalg as sp_linalg
from typing import Callable
import json
class InvalidIntervalError(Exception):
    """Exception raised for errors in the input interval."""
    pass
class InvalidMapAndWeightFunctionNumberError(Exception):
    """Exception raised for errors in the input interval."""
    pass

class SincMethod1d:
    __A: np.matrix
    methodType: str
    __predefinedMapAndWeightFunctionsAmount: int
    __nodes: np.array

    def __init__(self, methodType: str):
        self.methodType = methodType

    def setOperatorFunctions(self, mu: Callable, v: Callable, dmu:Callable, sigma: Callable):
        """
        From Stenger 1979:
        Lf(x) = f''(x) + mu(x)f'(x) + v(x)f(x) - sigma(x) = 0
        with f(a) = f(b) = 0, where (a,b) is an interval: (-inf, inf), (0, inf), (-1, 1)
        """
        self.__mu = mu
        self.__v = v
        self.__dmu = dmu
        self.__sigma = sigma

    def _is_interval_valid(self, interval):
        return (interval[0] == 0 and interval[1] == np.inf)

    def setParameters(self, a, b, h:float, N: int, mapAndWeightFunctionNumber: int,
                      customMapFunction: [Callable, Callable, Callable] = None, customWeightFunction: Callable = None):
        """
        Arguments:
                (a,b): one of the following intervals (-inf, inf), (0, inf), (-1, 1)
                h: step size of Sinc basis functions in reference space
                N: approximation (truncation) order
                mapAndWeightFunctionNumber: one of the predefined function sets for method,
                    in case 0 customFunctions are used
                customMapFunction: consists of [function, first derivative, second derivative]
                customWeightFunction: weight function in integrals of the method
        For predefined maps and weights the list of functionals,
         that is later used in calculations is also defined.
        The functional is of the form F[x, r(x), g(x), phi(x)]
         and the result is function at points w, x,
          where in approximation x_k are substituted into x and k * h into w
         F0: r * g
         F1: r * g / phi'
         F2: (r * g)'/phi'
         F3: (r * g)''/phi'
         F4: 2 * (r * g)' + r * g * phi''/phi'
         F5: r * g * phi'
        list of mapAndWeightFunctionNumber values:
            1:
                phi(z) = ln(sinh(z)), phi'(z) = coth(z), phi''(z) = -csch(z)^2, g(x) = 1/coth(z)
                F0 = 1/sqrt(1 + e^(-2w)) * r(x)
                F1 = 1/(1 + e^(-2w) * r(x)
                F2 = 1/sqrt(1 + e^(-2w)) * r'(x) + e^(-2w)/(1 + e^(-2w))^(3/2) * r(x)
                F3 = 1/(1 + e^(-2w)) * r''(x) + 2e^(-2w)/(1 + e^(-2w))^(3/2) * r'(x) - 2e^(-2w)/(1 + e^(-2w))^2 * r(x)
                F4 = 2/sqrt(1 + e^(-2w)) r'(x) + e^(-2w)/(1 + e^(-2w)) r(x)
                F5 = r(x)
        """
        self.__predefinedMapAndWeightFunctionsAmount = 1

        self.__interval = np.array([a, b], dtype=float)
        self.__h = h
        self.__approxOrder = 2 * N + 1
        self.__N = N
        if not self._is_interval_valid(self.__interval):
            raise InvalidIntervalError("Interval must be (0, infinity)")

        if (mapAndWeightFunctionNumber <= 0) or (mapAndWeightFunctionNumber > self.__predefinedMapAndWeightFunctionsAmount):
            if mapAndWeightFunctionNumber == 0:
                raise InvalidMapAndWeightFunctionNumberError("custom Map and Weight functions are not implemented")
            raise InvalidMapAndWeightFunctionNumberError("Map and Weight function number is incorrect")

        match mapAndWeightFunctionNumber:
            case 1:
                self.__map = lambda x: np.log(np.sinh(x))
                self.__dmap = lambda x: 1.0 / np.tanh(x)
                self.__d2map = lambda x: -1.0/np.sinh(x)**2
                self.__g = lambda x: np.tanh(x)

                rr = np.arange(start=-N, stop=N + 1, step=1)
                h = self.__h

                self.__nodes = np.log(np.exp(rr * h) + np.sqrt(1.0 + np.exp(2 * rr * h)))

                self.__F0 = lambda w, x, r: 1.0/np.sqrt(1.0 + np.exp(-2.0 * w)) * r(x)
                self.__F1 = lambda w, x, r: 1.0/(1.0 + np.exp(-2.0 * w)) * r(x)
                self.__F2 = lambda w, x, r, dr: 1.0/(1.0 + np.exp(-2.0 * w)) * dr(x) + \
                    np.exp(-2.0 * w)/(1.0 + np.exp(-2.0 * w))**(3.0/2.0) * r(x)
                self.__F3 = lambda w, x, r, dr, d2r: 1.0/(1.0 + np.exp(-2.0 * w)) * d2r(x) + \
                    2.0 * np.exp(-2.0 * w)/(1.0 + np.exp(-2.0 * w))**(3.0/2.0) * dr(x) + \
                    (-2.0) * np.exp(-2.0 * w)/(1.0 + np.exp(-2.0 * w))**2 * r(x)
                self.__F4 = lambda w, x, r, dr: 2.0/np.sqrt(1.0 + np.exp(-2.0 * w)) * dr(x) + \
                    np.exp(-2.0 * w)/(1.0 + np.exp(-2.0 * w)) * r(x)
                self.__F5 = lambda w, x, r: r(x)

    def calculateMatrix(self):
        """
            LE method type consider problem of the form Lf=0;

            EIG method type consider problem of the form Lf = lambda f, where
                there are multiple solutions, which are pairs of (eigenvalue, eigenvector)
        """
        if self.methodType == "LE":
            self.__calculateMatrix_LE()
        if self.methodType == "EIG":
            self.__calculateElements_EIG()


    def __calculateMatrix_LE(self):
        """
        Calculates matrix of discretized linear problem using formulas from Stenger 1979.

        o -- stands for composition of functions, phi(x) is a transformation from D to D_d
        dm_k, d2m_k, idm_k -- denotes phi'(x_k), phi''(x_k), 1/phi'(x_k) respectively.
        d^0_jk = {j==k: 1, j!=k: 0}
        d^1_jk = {j==k: 0, j!=k: (-1)^(k-j)/(k - j)}
        d^2_jk = {j==k: -pi^2/3, j!=k: -2 * (-1)^(k - j) / (k - j)^2}

            The discretization formulas and related matrices are listed below:
        F = int g(x) sigma(x) S(k, h) o phi(x) dx ≈ h g_k sigma_k * idm_k

        I = int g(x) mu(x) f(x) S(k, h) o phi(x) dx ≈ h g_k mu_k * idm_k * f_k

        D1 = int g(x) v(x) f'(x) S(k, h) o phi(x) dx ≈ h g_k v_k * idm_k * f'_k ≈
            - h sum_{j=-N}^N f_j [(v * g)'_j * idm_j * d^0_kj + (v * g)_j d^1_kj / h ]

        D2 = int g(x) v(x) f''(x) S(k, h) o phi(x) dx ≈ ≈ h g_k v_k * idm * f''_k ≈
            h sum_{j=-N}^N f_j [g''_j * idm_j * d^0_kj +
                                (2 * g'_j + g_j * d2m_j * idm_j) d^1_kj / h + g_j dm_j d^2_kj / h^2]
        """

        h = self.__h
        rangeN = np.arange(start=-self.__N, stop=self.__N + 1, step=1)

        jj, kk = np.meshgrid(rangeN, rangeN)
        I1m = (-1) ** (np.abs(jj - kk)) / (jj - kk)
        I1m = np.nan_to_num(I1m, posinf=0.0)
        I2m = -2.0 * (-1) ** (np.abs(jj - kk)) / (jj - kk) ** 2
        I2m = np.nan_to_num(I2m, neginf=-np.pi**2/3.0)

        w = h * rangeN
        x = self.__nodes.copy()
        F = h * self.__F1(w, x, self.__sigma)
        I = h * np.diag(self.__F1(w, x, self.__v))

        idFunc = lambda x: x * 0 + 1.0
        idFuncD = lambda x: 0.0
        idFunc2D = lambda x: 0.0

        D1 = -(h * np.diag(self.__F2(w, x, self.__mu, self.__dmu)) +
                   I1m @ np.diag(self.__F0(w, x, self.__mu)))

        D2 = (np.diag(self.__F3(w, x, idFunc, idFuncD, idFunc2D) * h) +
              I1m @ np.diag(self.__F4(w, x, idFunc, idFuncD)) +
              I2m @ np.diag(self.__F5(w, x, idFunc)/h))

        self.__A = I + D1 + D2
        self.__F = F

    def __calculateElements_EIG(self):
        """

        """
        h = self.__h
        rangeN = np.arange(start=-self.__N, stop=self.__N + 1, step=1)

        jj, kk = np.meshgrid(rangeN, rangeN)
        I1m = (-1) ** (np.abs(jj - kk)) / (jj - kk)
        I1m = np.nan_to_num(I1m, posinf=0.0)
        I2m = -2.0 * (-1) ** (np.abs(jj - kk)) / (jj - kk) ** 2
        I2m = np.nan_to_num(I2m, neginf=-np.pi ** 2 / 3.0)

        w = h * rangeN
        x = self.__nodes.copy()
        idFunc = lambda x: x * 0 + 1.0
        idFuncD = lambda x: 0.0
        idFunc2D = lambda x: 0.0
        lhsI = h * np.diag(self.__F1(w, x, self.__v))
        rhsI = h * np.diag(self.__F1(w, x, self.__sigma))


        D1 = -(h * np.diag(self.__F2(w, x, self.__mu, self.__dmu)) +
               I1m @ np.diag(self.__F0(w, x, self.__mu)))

        D2 = (np.diag(self.__F3(w, x, idFunc, idFuncD, idFunc2D) * h) +
              I1m @ np.diag(self.__F4(w, x, idFunc, idFuncD)) +
              I2m @ np.diag(self.__F5(w, x, idFunc) / h))

        self.__A = lhsI + D1 + D2
        self.__B = rhsI

    def solveEIG_denseMatrix(self, realize=True):
        """Only for single-domain case.
           Only for zero Dirichlet or Neumann boundary conditions
            Solves generalized eigenvalue problem:
            lhsMatrixElements @ u = lambda * rhsMatrixElements @ u

            Returns:
                ALL eigenpairs"""
        A = self.__A
        B = self.__B

        if sp_linalg.issymmetric(A) and sp_linalg.issymmetric(B):
            values, vectors = sp_linalg.eigh(A, B)
        else:
            values, vectors = sp_linalg.eig(A, B)
        if realize:
            eigvalIndices = np.argsort(np.real(values))
            values = np.real(values[eigvalIndices])
            vectors = np.real(vectors[:, eigvalIndices])
        self.solution = vectors[:, 0]

        return values, vectors

    def solveSLAE(self):
        """
        Solves system A @ u = F
        """
        A = self.__A.copy()
        F = self.__F.copy()
        solution = sp_linalg.solve(A, F)
        return solution

    def getMeshPoints(self):
        """

        """
        return self.__nodes




