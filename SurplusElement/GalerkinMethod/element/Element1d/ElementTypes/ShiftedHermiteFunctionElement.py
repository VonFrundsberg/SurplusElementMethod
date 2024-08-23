import numpy as np
import scipy.special as sp_spec
from numpy.polynomial.hermite import hermval
from numpy.polynomial.hermite import hermder
from SurplusElement.GalerkinMethod.element.Element1d.AuxClasses import ApproximationSpaceParameter as approxParam


class InvalidIntervalError(Exception):
    """Exception raised for errors in the input interval."""
    pass
class InvalidApproxOrderError(Exception):
    """Exception raised for errors in the approximation order."""
    pass

class ShiftedHermiteFunctionElement:
    """
    These are non-mapped elements with the interval (0, inf)
    The general form of basis functions is e^(-x*x/2) * (H_n(x) - shift)
    They are equal to: 1 - shift at 0; 0 at infinity
    Uses RationalInfHalfSpace mapping for (-1, 1) <-> (0, inf) coordinate transformations
    """
    def _is_interval_valid(self, interval):
        return (interval[0] == 0 and interval[1] == np.inf)

    def __init__(self, interval, parameters: approxParam, approxOrder: int):
        if not self._is_interval_valid(interval):
            raise InvalidIntervalError("Interval must be (0, infinity)")

        if not isinstance(approxOrder, int) or approxOrder <= 0:
            raise InvalidApproxOrderError("Approximation order must be a positive integer")

        self.interval = np.array(interval, dtype=float)
        self.approxOrder = int(approxOrder)
        self.shift = parameters.shift
        self.s = parameters.s
        L = 1.0
        self.map = lambda x: (L * (1.0 + x) / (1.0 - x) + self.interval[0])
        self.inverseMap = lambda x: (-x + self.interval[0] + L) / (-x + self.interval[0] - L)

        self.derivativeMap = lambda x: (x - 1) ** 2 / (2 * L)
        self.inverseDerivativeMap = lambda x: 2 * L / (x - 1) ** 2
        self.zeroHermVal = hermval(0, np.eye(self.approxOrder))
        kk = np.arange(start=0, stop=self.approxOrder, step=1)
        self.scaleVector = 1.0/np.sqrt(2**kk * sp_spec.factorial(kk) * np.sqrt(np.pi))

    def eval(self, x):
        """ Evaluates basis functions at points x

                    Arguments:
                        x: evaluation points

                    Returns:
                        result: array with the shape: (*x.shape, degree of element)
                """
        # x = np.array([0.0, 0.5, 1.0, 1.5, 100.0], dtype=float)
        # print(self.approxOrder)
        # nRange = np.arange(start=0, stop=self.approxOrder, step=1)
        # normalization = np.sqrt(2**nRange * sp_spec.factorial(nRange))
        # time.sleep(500)
        # print(((np.exp(-x*x/2.0)*((hermval(x, np.eye(self.approxOrder)).T - hermval(0, np.eye(self.approxOrder))).T
        #                           - self.shift)).T).shape)
        hermValT = hermval(x, np.eye(self.approxOrder)).T
        return ((np.exp(-x*x/(self.s * 2.0))*
                ((hermValT - self.zeroHermVal).T
                                  - self.shift)).T) * self.scaleVector

    def evaluateExpansion(self, coefficients: list[float], x: list[float]):
        """ Evaluates expansion in basis functions with given coefficients at points x
                            Arguments:
                                coefficients: list of basis coefficients
                                x: list of evaluation points

                            Returns:
                                result: array with the shape: (*x.shape)
                """
        hermValT = hermval(x, np.eye(self.approxOrder)).T
        evaluatedBasis =  ((np.exp(-x * x / (self.s * 2.0)) *
                           ((((hermValT - self.zeroHermVal).T))
                 - self.shift)).T) * self.scaleVector

        return evaluatedBasis @ coefficients
    def evalDiff(self, x):
        """ Evaluates derivatives of basis functions at points x

            Arguments:
                x: one-dimensional array of evaluation points

            Returns:
                result: array with the shape:  (*x.shape, degree of element)
        """
        # np.set_printoptions(precision=4, suppress=True)
        # x = np.array([0.0, 0.5, 1.0, 1.5, 100.0], dtype=float)
        derivative = hermder(np.eye(self.approxOrder), m=1)
        derivativeValues = hermval(x, derivative)
        hermiteValuesT = hermval(x, np.eye(self.approxOrder)).T
        # print(derivativeValues.T)
        # print(laguerreValues.T)
        # self.shift = 1.0
        # print(-((0.5*np.exp(-x/2.0)*(-self.shift + laguerreValues - 2 * derivativeValues))).T)
        # time.sleep(500)
        # print(((((np.exp(-x*x/2.0)*(self.shift - x * ((hermiteValues.T - zeroHermiteValues).T) + derivativeValues)))).T).shape)
        return ((1.0/self.s)*(((np.exp(-x*x/(self.s * 2.0))*
                  (((x*(self.shift - ((hermiteValuesT - self.zeroHermVal).T))))
                   + self.s*derivativeValues)))).T) * self.scaleVector