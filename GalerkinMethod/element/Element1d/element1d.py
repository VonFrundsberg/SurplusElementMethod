import numpy as np
from mathematics import spectral as spec
from enum import Enum
from GalerkinMethod.element.Element1d.ElementTypes.PolyAffineElement import PolyAffineElement as PolyAffine
from GalerkinMethod.element.Element1d.ElementTypes.PeriodicCardinalElement import PeriodicCardinalElement as PeriodicCardinal
from GalerkinMethod.element.Element1d.ElementTypes.RationalInfHalfSpaceElement import RationalInfHalfSpaceElement as RationalInfHalfSpace
from GalerkinMethod.element.Element1d.ElementTypes.ExponentialInfHalfSpace import ExponentialInfHalfSpaceElement as ExponentialInfHalfSpace
class ElementType(Enum):
    LINEAR = 0
    RATIONAL_INF_HALF_SPACE = 1
    EXPONENTIAL_INF_HALF_SPACE = 2
    PERIODIC_CARDINAL = 3

class Element1d:
    def __getElementTypeInstance(self):
        match self.elementType:
            case ElementType.LINEAR.value:
                return PolyAffine(self.interval, self.approxOrder, self.dirichletBoundaryConditions)
            case ElementType.RATIONAL_INF_HALF_SPACE.value:
                return RationalInfHalfSpace(self.interval, self.approxOrder, self.dirichletBoundaryConditions)
            case ElementType.EXPONENTIAL_INF_HALF_SPACE.value:
                return ExponentialInfHalfSpace(self.interval, self.approxOrder, self.dirichletBoundaryConditions)
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

        self.eval = self.__elementTypeInstance.eval
        self.evalDiff = self.__elementTypeInstance.evalDiff
        if hasattr(self.__elementTypeInstance, 'evalDiffRefNodes'):
            self.evalDiffRefNodes = self.__elementTypeInstance.evalDiffRefNodes

        if hasattr(self.__elementTypeInstance, 'evaluateExpansion'):
            self.evaluateExpansion = self.__elementTypeInstance.evaluateExpansion
