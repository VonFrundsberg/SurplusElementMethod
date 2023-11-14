import numpy as np
from mathematics import spectral as spec
class PolyAffineElement:
    def __init__(self, interval, approxOrder, dirichletBoundaryConditions=None):
        self.interval = np.array(interval)
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
            for it in self.dirichletBoundaryConditions:
                self.refPointVal[it[0], it[0]] = it[1]

        self.refPointDiffVal = \
            spec.chebDiffMatrix(self.approxOrder, a=-1, b=1).\
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