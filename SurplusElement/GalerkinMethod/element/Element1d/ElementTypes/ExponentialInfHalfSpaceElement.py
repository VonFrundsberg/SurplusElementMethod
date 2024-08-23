import numpy as np
from SurplusElement.mathematics import spectral as spec
from SurplusElement.GalerkinMethod.element.Element1d.AuxClasses import ApproximationSpaceParameter as approxParam

class InvalidIntervalError(Exception):
    """Exception raised for errors in the input interval."""
    pass
class InvalidApproxOrderError(Exception):
    """Exception raised for errors in the approximation order."""
    pass

class ExponentialInfHalfSpaceElement:
    def _is_interval_valid(self, interval):
        return (interval[0] == 0 and interval[1] == np.inf)

    def __init__(self, interval, approxOrder, dirichletBoundaryConditions=None, parameters: approxParam = None):

        if not self._is_interval_valid(interval):
            raise InvalidIntervalError("Interval must be (0, infinity)")

        if not isinstance(approxOrder, int) or approxOrder <= 0:
            raise InvalidApproxOrderError("Approximation order must be a positive integer")

        self.interval = np.array(interval, dtype=float)
        self.approxOrder = int(approxOrder)
        self.dirichletBoundaryConditions = dirichletBoundaryConditions
        self.parameters = parameters

        if parameters is not None:
            self.s = parameters.s

        s = self.s

        self.L_s = np.sinh(s)
        # print(self.L_s)

        if self.interval[1] == np.inf:
            self.map = lambda x: np.sinh(0.5 * s * (1.0 + x))
            self.inverseMap = lambda x: 2.0/s * np.log(x + np.sqrt(x*x + 1.0)) - 1.0

            self.derivativeMap = lambda x: 2.0/(s * np.cosh(0.5 * s * (1.0 + x)))
            self.inverseDerivativeMap = lambda x: 0.5 * s * np.cosh(0.5 * s * (1.0 + x))


        self.refPointVal = np.eye(self.approxOrder)
        if dirichletBoundaryConditions is not None:
            for bc in dirichletBoundaryConditions:
                if bc.boundaryPoint == self.interval[0]:
                    self.refPointVal[0, 0] = bc.boundaryValue
                if bc.boundaryPoint == self.interval[1]:
                    self.refPointVal[-1, -1] = bc.boundaryValue



        self.refPointDiffVal = spec.chebDiffMatrix(self.approxOrder, a=-1, b=1) @ self.refPointVal

    def getRefNodes(self):
        chebNodes = spec.chebNodes(self.approxOrder, a=-1.0, b=1.0)
        return self.map(chebNodes)
    def evaluateExpansion(self, coefficients: list[float], x: list[float]):
        """ Evaluates expansion in basis functions with given coefficients at points x
                            Arguments:
                                coefficients: list of basis coefficients
                                x: list of evaluation points

                            Returns:
                                result: array with the shape: (*x.shape)
                """
        infValues = np.isinf(x)
        x = self.inverseMap(np.atleast_1d(x))
        x[infValues] = 1.0
        basisMatrix = np.eye(self.approxOrder)

        evaluatedBasis = spec.barycentricChebInterpolate(basisMatrix, x, a=-1.0, b=1.0, axis=0)

        return evaluatedBasis @ coefficients

    def eval(self, x):
        """ Evaluates basis functions at points x

                    Arguments:
                        x: evaluation points

                    Returns:
                        result: array with the shape: (*x.shape, degree of element)
                """
        infValues = np.isinf(x)
        x = self.inverseMap(np.atleast_1d(x))
        x[infValues] = 1.0
        basisMatrix = self.refPointVal
        return spec.barycentricChebInterpolate(basisMatrix, x, a=-1.0, b=1.0, axis=0)

    def evalDiff(self, x):
        """ Evaluates derivatives of basis functions at points x

            Arguments:
                x: one-dimensional array of evaluation points

            Returns:
                result: array with the shape:  (approximation order of element, len(x))
        """

        infValues = np.isinf(x)
        x = self.inverseMap(np.atleast_1d(x))
        x[infValues] = 1.0
        derivativeBasisMatrix = self.refPointDiffVal
        jacobian = self.derivativeMap(x)
        return spec.barycentricChebInterpolate(f=derivativeBasisMatrix, x=x, a=-1.0, b=1.0, axis=0) \
                     * np.reshape(jacobian, (*x.shape, 1))