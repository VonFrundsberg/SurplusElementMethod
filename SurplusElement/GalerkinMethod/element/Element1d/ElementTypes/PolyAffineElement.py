import numpy as np
from SurplusElement.mathematics import spectral as spec
from SurplusElement.GalerkinMethod.element.Element1d.AuxClasses import DirichletBoundaryCondition
class PolyAffineElement:
    """"
    Basis functions are Lagrange polynomials based on Chebyshev points of second kind:
    x_j = -cos(j * pi / n), j = 0,...,n
    "native" interval is [-1, 1], if another interval [a, b] is specified (with finite values a, b),
     the polynomials are mapped using linear transformation q * x + p, where
     q = (b - a)/2, p = (b + a)/2.
     Dirichlet boundary conditions can be specified only at a, b
    """
    def __init__(self, interval, approxOrder, dirichletBoundaryConditions : list[DirichletBoundaryCondition]=None):
        self.interval = np.array(interval, dtype=float)
        self.approxOrder = int(approxOrder)
        self.dirichletBoundaryConditions = dirichletBoundaryConditions

        a = self.interval[0]; b = self.interval[1]
        q = (b - a) / 2.0;        p = (b + a) / 2.0
        self.map = lambda x: (q * x + p)
        self.inverseMap = lambda x: (x - p) / q

        self.derivativeMap = lambda x: 1.0 / (q + x * 0)
        self.inverseDerivativeMap = lambda x: q + 0
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
        x = np.atleast_1d(x)
        basisMatrix = np.eye(self.approxOrder)
        evaluatedBasis = spec.barycentricChebInterpolate(basisMatrix, x, a=self.interval[0], b=self.interval[1], axis=0)

        return evaluatedBasis @ coefficients

    def eval(self, x):
        """ Evaluates basis functions at points x

                    Arguments:
                        x: evaluation points

                    Returns:
                        result: array with the shape: (*x.shape, degree of element)
        """
        x = np.atleast_1d(x)
        basisMatrix = self.refPointVal
        # if x.size == 1:
        #     if x[0] == self.interval[0]:
        #         return np.array([basisMatrix[0, :]])
        #     if x[0] == self.interval[-1]:
        #         return np.array([basisMatrix[-1, :]])

        return spec.barycentricChebInterpolate(basisMatrix, x, a=self.interval[0], b=self.interval[1], axis=0)

    def evalDiff(self, x):
        """ Evaluates derivatives of basis functions at points x

            Arguments:
                x: one-dimensional array of evaluation points

            Returns:
                result: array with the shape:  (approximation order of element, len(x))
        """
        x = np.atleast_1d(x)
        derivativeBasisMatrix = self.refPointDiffVal
        # if x.size == 1:
        #     if x[0] == self.interval[0]:
        #             return self.derivativeMap(-1)*np.array([derivativeBasisMatrix[0, :]])
        #     if x[0] == self.interval[-1]:
        #             return self.derivativeMap(1)*np.array([derivativeBasisMatrix[-1, :]])
        # print(self.derivativeMap(x))
        return spec.barycentricChebInterpolate(f=derivativeBasisMatrix,
                    x=x, a=self.interval[0], b=self.interval[1], axis=0) \
                     * np.reshape(self.derivativeMap(x), (*x.shape, 1))