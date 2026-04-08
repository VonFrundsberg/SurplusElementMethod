import numpy as np
from SurplusElement.mathematics import spectral as spec
from SurplusElement.GalerkinMethod.element.Element1d.AuxClasses import ApproximationSpaceParameter as approxParam
import time as time
class RationalInfHalfSpaceElement:
    def __init__(self, interval, parameters:approxParam, approxOrder, dirichletBoundaryConditions=None):
        self.interval = np.array(interval, dtype=float)
        self.approxOrder = int(approxOrder)
        self.dirichletBoundaryConditions = dirichletBoundaryConditions

        if self.interval[0] == -np.inf:
            self.map = lambda x: -((1.0 - x) / (1.0 + x) - self.interval[1])
            self.inverseMap = lambda x: (x + 1.0 - self.interval[1]) / (-x + self.interval[1] + 1.0)
            self.derivativeMap = lambda x: (x + 1) ** 2 / 2
            self.derivativeMap2 = lambda x: x + 1
            self.inverseDerivativeMap = lambda x: 2 / (x + 1) ** 2
            return
        self.s = parameters.s
        s = self.s
        if self.interval[1] == np.inf:
            # L = self.interval[0]
            self.map = lambda x: (s*(1.0 + x) / (1.0 - x) + self.interval[0])
            self.inverseMap = lambda x: (-x + self.interval[0] + s) / (-x + self.interval[0] - s)
            self.derivativeMap = lambda x: (x - 1) ** 2 / (2 * s)
            self.derivativeMap2 = lambda x: (x - 1) / (s)
            self.inverseDerivativeMap = lambda x: 2 * s / (x - 1) ** 2

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

    def evaluateExpansionDerivatives(self,
                                     coefficients: list[float],
                                     x: list[float],
                                     derivative: int = 1):
        """  """
        x = np.atleast_1d(x)
        chebNodes = spec.chebNodes(self.approxOrder, a=-1.0, b=1.0)
        if derivative == 1:
            basisMatrix = spec.barycentricChebInterpolate(f=self.refPointDiffVal,
                                x=chebNodes, a=-1.0, b=1.0, axis=0) \
                          * np.reshape(self.derivativeMap(chebNodes), (*chebNodes.shape, 1))
            evaluatedBasis = spec.barycentricChebInterpolate(basisMatrix, x, a=self.interval[0], b=self.interval[1],
                                                             axis=0)
            result = evaluatedBasis @ coefficients
            return result
        if derivative == 2:
            basisMatrixL = spec.barycentricChebInterpolate(f=(self.refPointDiffVal @ self.refPointDiffVal),
                                x=chebNodes, a=-1.0, b=1.0, axis=0) \
                           * np.reshape(self.derivativeMap(chebNodes)**2, (*chebNodes.shape, 1))
            basisMatrixR = spec.barycentricChebInterpolate(f=self.refPointDiffVal,
                                    x=chebNodes, a=-1.0, b=1.0, axis=0) \
                          * np.reshape(self.derivativeMap2(chebNodes), (*chebNodes.shape, 1))
            evaluatedBasisL = spec.barycentricChebInterpolate(basisMatrixL, x, a=self.interval[0], b=self.interval[1],
                                                             axis=0)
            evaluatedBasisR = spec.barycentricChebInterpolate(basisMatrixR, x, a=self.interval[0], b=self.interval[1],
                                                             axis=0)
            return evaluatedBasisL @ coefficients + evaluatedBasisR @ coefficients
        if derivative >= 3:
            raise Exception("evaluateExpansionDerivatives: only derivatives up to second order are allowed")

    def eval(self, x):
        """ Evaluates basis functions at points x

                    Arguments:
                        x: evaluation points

                    Returns:
                        result: array with the shape: (*x.shape, degree of element)
                """
        infValues = np.isinf(x)
        outOfRangeLeft = np.argwhere(x < self.interval[0])
        outOfRangeRight = np.argwhere(x > self.interval[1])
        x = self.inverseMap(np.atleast_1d(x))
        x[infValues] = 1.0
        """-1.1 and 1.1 are here just to make the output of barycentric interpolate to be zero on these values"""
        x[outOfRangeLeft] = -1.1 * np.ones(x[outOfRangeLeft].shape)
        x[outOfRangeRight] = 1.1 * np.ones(x[outOfRangeLeft].shape)
        x[infValues] = 1.0
        basisMatrix = self.refPointVal
        # if x.size == 1:
        #     if x[0] == self.interval[0]:
        #         return np.array([basisMatrix[0, :]])
        #     if x[0] == self.interval[-1]:
        #         return np.array([basisMatrix[-1, :]])

        return spec.barycentricChebInterpolate(basisMatrix, x, a=-1.0, b=1.0, axis=0)

    def evalDiff(self, x):
        """ Evaluates derivatives of basis functions at points x

            Arguments:
                x: one-dimensional array of evaluation points

            Returns:
                result: array with the shape:  (approximation order of element, len(x))
        """
        x = np.atleast_1d(x)
        infValues = np.isinf(x)
        outOfRangeLeft = np.argwhere(x < self.interval[0])
        outOfRangeRight = np.argwhere(x > self.interval[1])
        x = self.inverseMap(np.atleast_1d(x))
        x[infValues] = 1.0
        """-1.1 and 1.1 are here just to make the output of barycentric interpolate to be zero on these values"""
        x[outOfRangeLeft] = -1.1 * np.ones(x[outOfRangeLeft].shape)
        x[outOfRangeRight] = 1.1 * np.ones(x[outOfRangeLeft].shape)
        derivativeBasisMatrix = self.refPointDiffVal
        jacobian = self.derivativeMap(x)
        # print("jacobian", jacobian)
        # if x.size == 1:
        #     if x[0] == self.interval[0]:
        #             return self.derivativeMap(-1)*np.array([derivativeBasisMatrix[0, :]])
        #     if x[0] == self.interval[-1]:
        #             return self.derivativeMap(1)*np.array([derivativeBasisMatrix[-1, :]])
        return spec.barycentricChebInterpolate(f=derivativeBasisMatrix, x=x, a=-1.0, b=1.0, axis=0) \
                     * np.reshape(jacobian, (*x.shape, 1))