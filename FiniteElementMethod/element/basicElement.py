import numpy as np
from mathematics import spectral as sp

class basicElement():
    def __init__(self, interval, polynomialOrder, mappingType, boundaryConditions=[]):
        """Constructor of basic (one-dimensional) element

        Arguments:
            interval:
            polynomialOrder:
            mappingType: 0 is linear qx+p, 1 is rational (1+x)/(1-x), 2 is exponential ???
            boundaryConditions:

        Returns: None
        """
        self.interval = np.array(interval)
        self.polynomialOrder = polynomialOrder
        self.boundaryConditions = boundaryConditions
        self.refPointVal = np.eye(polynomialOrder)
        for it in self.boundaryConditions:
            self.refPointVal[it[0], it[0]] = it[1]

        self.refPointDiffVal = \
            sp.ChebDiffMatrix(self.polynomialOrder, a=interval[0], b=interval[1]).\
                dot(self.refPointVal)
        match mappingType:
            case 0:
                a = self.interval[0]; b = self.interval[1]
                q = (b - a) / 2.0; p = (b + a) / 2.0
                self.map = lambda x: (q * x + p)
                self.inverseMap = lambda x: (x - p)/q

                self.derivativeMap = lambda x: 1.0 / (q + x * 0)
                self.inverseDerivativeMap = lambda x: q + 0
            case 1:
                if self.interval[0] == -np.inf:
                    self.map = lambda x: -((1.0 - x) / (1.0 + x) - self.interval[1])
                    self.inverseMap = lambda x: (x + 1.0 - self.interval[1])/(-x + self.interval[1] + 1.0)

                    self.derivativeMap = lambda x: (x + 1) ** 2 / 2
                    self.inverseDerivativeMap = lambda x: 2/(x + 1)**2
                    return

                if self.interval[1] == np.inf:
                    self.map = lambda x: ((1.0 + x) / (1.0 - x) + self.interval[0])
                    self.inverseMap = lambda x: (-x + self.interval[0] + 1.0) / (-x + self.interval[0] - 1.0)

                    self.derivativeMap = lambda x: (x - 1) ** 2 / 2
                    self.inverseDerivativeMap = lambda x: 2/(x - 1) ** 2
                    return
            case 2:
                if self.interval[0] == -np.inf:
                    self.map = lambda x: -((1.0 - x) / (1.0 + x) - self.interval[1])
                    self.inverseMap = lambda x: (x + 1.0 - self.interval[1])/(-x + self.interval[1] + 1.0)

                    self.derivativeMap = lambda x: (x + 1) ** 2 / 2
                    self.inverseDerivativeMap = lambda x: 2/(x + 1)**2
                    return

                if self.interval[1] == np.inf:
                    self.map = lambda x: np.log((1.0 + x) / (1.0 - x) + 1) + self.interval[0]
                    self.inverseMap = lambda x: np.exp(self.interval[0] - x)*(np.exp(x - self.interval[0]) - 2.0)

                    self.derivativeMap = lambda x: (1.0 - x)
                    self.inverseDerivativeMap = lambda x: 1/(1.0 - x)
                    return
    def evalAtChebPoints(self):
        return self.refPointVal
    def evalDiffAtChebPoints(self):
        return self.refPointDiffVal
    def evaluatePoints(self, x):
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

        result = sp.barycentricChebInterpolate(basisMatrix, x, a=-1, b=1)
        return result
    def getMappedRefPoints(self):
        x = sp.calcChebPoints(self.polynomialOrder, -1, 1)
        x = self.map(x)
        return x
    def evalDiff(self, x):
        """ Evaluates derivatives of basis functions at points x

            Arguments:
                x: evaluation points

            Returns:
                result: array with the shape: (*x.shape, degree of element)
        """
        x = np.atleast_1d(x)
        derivativeBasisMatrix = self.refPointDiffVal
        if x.size == 1:
            if x[0] == self.interval[0]:
                return self.derivativeMap(-1)*np.array([derivativeBasisMatrix[0, :]])
            if x[0] == self.interval[-1]:
                return self.derivativeMap(1)*np.array([derivativeBasisMatrix[-1, :]])
        # xs = np.array(self.imap(x), dtype=np.float)
        result = sp.barycentricChebInterpolate(f=derivativeBasisMatrix, x=x)\
                 *np.reshape(self.derivativeMap(x), (*x.shape, 1))
        return result
    def generateFunction(self):
        return lambda x: self.evaluatePoints(x)
    def generateDerivativeFunction(self):
        return lambda x: self.evalDiff(x)