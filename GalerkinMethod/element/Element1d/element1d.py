import numpy as np
from mathematics import spectral as spec
from enum import Enum
from GalerkinMethod.element.Element1d.PolyAffineElement import PolyAffineElement as PolyAffine
from GalerkinMethod.element.Element1d.PeriodicCardinalElement import PeriodicCardinalElement as PeriodicCardinal

class ElementType(Enum):
    LINEAR = 0
    RATIONAL_INF_HALF_SPACE = 1
    EXPONENTIAL_INF_HALF_SPACE = 2
    PERIODIC_CARDINAL = 3

class Element1d:
    def __initializeRATIONAL_INF_HALF_SPACE(self):
        if self.interval[0] == -np.inf:
            self.map = lambda x: -((1.0 - x) / (1.0 + x) - self.interval[1])
            self.inverseMap = lambda x: (x + 1.0 - self.interval[1]) / (-x + self.interval[1] + 1.0)

            self.derivativeMap = lambda x: (x + 1) ** 2 / 2
            self.inverseDerivativeMap = lambda x: 2 / (x + 1) ** 2
            return

        if self.interval[1] == np.inf:
            self.map = lambda x: ((1.0 + x) / (1.0 - x) + self.interval[0])
            self.inverseMap = lambda x: (-x + self.interval[0] + 1.0) / (-x + self.interval[0] - 1.0)

            self.derivativeMap = lambda x: (x - 1) ** 2 / 2
            self.inverseDerivativeMap = lambda x: 2 / (x - 1) ** 2

    def __initializeEXPONENTIAL_INF_HALF_SPACE(self):
        if self.interval[1] == np.inf:
            self.map = lambda x: np.log((1.0 + x) / (1.0 - x) + 1) + self.interval[0]
            self.inverseMap = lambda x: np.exp(self.interval[0] - x) * (np.exp(x - self.interval[0]) - 2.0)

            self.derivativeMap = lambda x: (1.0 - x)
            self.inverseDerivativeMap = lambda x: 1 / (1.0 - x)

    def __getElementTypeInstance(self):
        match self.elementType:
            case ElementType.LINEAR.value:
                return PolyAffine(self.interval, self.approxOrder, self.dirichletBoundaryConditions)
            case ElementType.RATIONAL_INF_HALF_SPACE.value:
                self.__initializeRATIONAL_INF_HALF_SPACE()
            case ElementType.EXPONENTIAL_INF_HALF_SPACE.value:
                self.__initializeRATIONAL_INF_HALF_SPACE()
            case ElementType.PERIODIC_CARDINAL.value:
                return PeriodicCardinal(self.interval, self.approxOrder, self.dirichletBoundaryConditions)
    def __init__(self, interval, approxOrder, elementType, dirichletBoundaryConditions=None):
        """Constructor of one-dimensional galerkin element

        Arguments:
            interval:
            approxOrder:
            elementType: 0 is linear qx+p, 1 is rational (1+x)/(1-x), 2 is exponential, 3 is periodic
            boundaryConditions:

        Returns: None
        """
        self.interval = np.array(interval)
        self.approxOrder = int(approxOrder)
        self.elementType = int(elementType)
        self.dirichletBoundaryConditions = dirichletBoundaryConditions

        self.refPointVal = np.eye(self.approxOrder)

        self.__elementTypeInstance = self.__getElementTypeInstance()
        self.map = self.__elementTypeInstance.map
        self.inverseMap = self.__elementTypeInstance.inverseMap
        self.derivativeMap = self.__elementTypeInstance.derivativeMap
        self.inverseDerivativeMap = self.__elementTypeInstance.inverseDerivativeMap

    def evalDiffRefNodes(self):
        if self.elemType < 3:
            return self.refPointDiffVal
        elif self.elemType == 3:
            return spec.periodicDiffMatrix(self.approxOrder)
        elif self.elemType == 4:
            return spec.periodicDiffMatrix(self.approxOrder, halfInterval=True)
    def eval(self, x):
        """ Evaluates basis functions at points x

            Arguments:
                x: evaluation points

            Returns:
                result: array with the shape: (*x.shape, degree of element)
        """
        return self.__elementTypeInstance.eval(x)

    def evalDiff(self, x):
        """ Evaluates derivatives of basis functions at points x

            Arguments:
                x: one-dimensional array of evaluation points

            Returns:
                result: array with the shape:  (approximation order of element, len(x))
        """
        return self.__elementTypeInstance.evalDiff(x)