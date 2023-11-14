import numpy as np
from mathematics import spectral as spec

class RationalInfHalfSpaceElement:
    def __init__(self, interval, approxOrder, dirichletBoundaryConditions=None):
        self.interval = np.array(interval)
        self.approxOrder = int(approxOrder)
        self.dirichletBoundaryConditions = dirichletBoundaryConditions

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
        self.refPointVal = np.eye(self.approxOrder)
        if dirichletBoundaryConditions is not None:
            for it in self.dirichletBoundaryConditions:
                self.refPointVal[it[0], it[0]] = it[1]

        self.refPointDiffVal = \
            spec.chebDiffMatrix(self.approxOrder, a=-1, b=1). \
                dot(self.refPointVal)

    def eval(self, x):
        """ Evaluates basis functions at points x

                    Arguments:
                        x: evaluation points

                    Returns:
                        result: array with the shape: (*x.shape, degree of element)
                """
        x = np.atleast_1d(x)
        basisMatrix = self.refPointVal
        if x.size == 1:
            if x[0] == self.interval[0]:
                return np.array([basisMatrix[0, :]])
            if x[0] == self.interval[-1]:
                return np.array([basisMatrix[-1, :]])

        return spec.barycentricChebInterpolate(basisMatrix, x, a=-1, b=1, axis=0)

    def evalDiff(self, x):
        """ Evaluates derivatives of basis functions at points x

            Arguments:
                x: one-dimensional array of evaluation points

            Returns:
                result: array with the shape:  (approximation order of element, len(x))
        """
        x = np.atleast_1d(x)
        derivativeBasisMatrix = self.refPointDiffVal
        if x.size == 1:
            if x[0] == self.interval[0]:
                    return self.derivativeMap(-1)*np.array([derivativeBasisMatrix[0, :]])
            if x[0] == self.interval[-1]:
                    return self.derivativeMap(1)*np.array([derivativeBasisMatrix[-1, :]])
        return spec.barycentricChebInterpolate(f=derivativeBasisMatrix, x=x, a=-1, b=1, axis=0) \
                     * np.reshape(self.derivativeMap(x), (*x.shape, 1))