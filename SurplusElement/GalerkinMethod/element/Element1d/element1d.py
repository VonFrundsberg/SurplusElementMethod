import numpy as np
from SurplusElement.GalerkinMethod.element.Element1d.AuxClasses import DirichletBoundaryCondition as DBC
from SurplusElement.GalerkinMethod.element.Element1d.AuxClasses import ApproximationSpaceParameter as ASP
from enum import Enum
from SurplusElement.GalerkinMethod.element.Element1d.ElementTypes.PolyAffineElement import PolyAffineElement as PolyAffine
from SurplusElement.GalerkinMethod.element.Element1d.ElementTypes.PeriodicCardinalElement import PeriodicCardinalElement as PeriodicCardinal
from SurplusElement.GalerkinMethod.element.Element1d.ElementTypes.RationalInfHalfSpaceElement import RationalInfHalfSpaceElement as RationalInfHalfSpace
from SurplusElement.GalerkinMethod.element.Element1d.ElementTypes.ExponentialInfHalfSpaceElement import ExponentialInfHalfSpaceElement as ExponentialInfHalfSpace
from SurplusElement.GalerkinMethod.element.Element1d.ElementTypes.ShiftedLaguerreFunctionElement import ShiftedLaguerreFunctionElement as ShiftedLaguerreFunction
from SurplusElement.GalerkinMethod.element.Element1d.ElementTypes.LogarithmicInfHalfSpaceElement import LogarithmicInfHalfSpaceElement as LogarithmicInfHalsSpace
from SurplusElement.GalerkinMethod.element.Element1d.ElementTypes.ShiftedHermiteFunctionElement import ShiftedHermiteFunctionElement as SchiftedHermiteFunction
class ElementType(Enum):
    LINEAR = 0
    RATIONAL_INF_HALF_SPACE = 1
    EXPONENTIAL_INF_HALF_SPACE = 2
    PERIODIC_CARDINAL = 3
    SHIFTED_LAGUERRE = 4
    LOGARITHMIC_INF_HALF_SPACE = 5
    SHIFTED_HERMITE = 6

class Element1d:
    def __getElementTypeInstance(self):
        match self.elementType:
            case ElementType.LINEAR.value:
                return PolyAffine(self.interval, self.approxOrder, self.dirichletBoundaryConditions)
            case ElementType.RATIONAL_INF_HALF_SPACE.value:
                return RationalInfHalfSpace(interval=self.interval, approxOrder=self.approxOrder,
                                            parameters=self.parameters, dirichletBoundaryConditions=self.dirichletBoundaryConditions)
            case ElementType.EXPONENTIAL_INF_HALF_SPACE.value:
                return ExponentialInfHalfSpace(interval=self.interval, approxOrder=self.approxOrder,
                                               dirichletBoundaryConditions=self.dirichletBoundaryConditions, parameters=self.parameters)
            case ElementType.PERIODIC_CARDINAL.value:
                return PeriodicCardinal(self.interval, self.approxOrder, self.dirichletBoundaryConditions)
            case ElementType.SHIFTED_LAGUERRE.value:
                return ShiftedLaguerreFunction(interval=self.interval, parameters=self.parameters, approxOrder=self.approxOrder)
            case ElementType.LOGARITHMIC_INF_HALF_SPACE.value:
                return LogarithmicInfHalsSpace(interval=self.interval, parameters=self.parameters, approxOrder=self.approxOrder, dirichletBoundaryConditions=self.dirichletBoundaryConditions)
            case ElementType.SHIFTED_HERMITE.value:
                return SchiftedHermiteFunction(interval=self.interval, parameters=self.parameters, approxOrder=self.approxOrder)

    def __init__(self, interval, approxOrder: int , elementType: int, dirichletBoundaryConditions: list[DBC], parameters: ASP):
        """Constructor of one-dimensional galerkin element
        Arguments:
            interval:
            approxOrder:
            elementType:
                0 - linear;
                1 - rational half space;
                2 - exponential half space;
                3 - periodic functions;
                4 - laguerre functions;
                5 - logarithmic half space;
            boundaryConditions:

        Returns: None
        """
        self.interval = np.array(interval)
        self.approxOrder = int(approxOrder)
        self.elementType = int(elementType)
        self.parameters = parameters
        self.dirichletBoundaryConditions = dirichletBoundaryConditions

        self.refPointVal = np.eye(self.approxOrder)

        self.__elementTypeInstance = self.__getElementTypeInstance()

        if hasattr(self.__elementTypeInstance, 'map'):
            self.map = self.__elementTypeInstance.map
        if hasattr(self.__elementTypeInstance, 'inverseMap'):
            self.inverseMap = self.__elementTypeInstance.inverseMap
        if hasattr(self.__elementTypeInstance, 'derivativeMap'):
            self.derivativeMap = self.__elementTypeInstance.derivativeMap
        if hasattr(self.__elementTypeInstance, 'inverseDerivativeMap'):
            self.inverseDerivativeMap = self.__elementTypeInstance.inverseDerivativeMap

        self.eval = self.__elementTypeInstance.eval
        self.evalDiff = self.__elementTypeInstance.evalDiff

        if hasattr(self.__elementTypeInstance, 'getRefNodes'):
            self.getRefNodes = self.__elementTypeInstance.getRefNodes

        if hasattr(self.__elementTypeInstance, 'evalDiffRefNodes'):
            self.evalDiffRefNodes = self.__elementTypeInstance.evalDiffRefNodes

        if hasattr(self.__elementTypeInstance, 'evaluateExpansion'):
            self.evaluateExpansion = self.__elementTypeInstance.evaluateExpansion
