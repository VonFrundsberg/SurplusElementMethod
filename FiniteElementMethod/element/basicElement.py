import numpy as np
from mathematics import spectral as spec

class basicElement():
    def __init__(self, interval, approxOrder, elemType, boundaryConditions=[]):
        """Constructor of basic (one-dimensional) element

        Arguments:
            interval:
            approxOrder:
            elemType: 0 is linear qx+p, 1 is rational (1+x)/(1-x), 2 is exponential ???
            boundaryConditions:

        Returns: None
        """
        self.interval = np.array(interval)
        self.approxOrder = approxOrder
        self.boundaryConditions = boundaryConditions
        self.refPointVal = np.eye(approxOrder)
        for it in self.boundaryConditions:
            self.refPointVal[it[0], it[0]] = it[1]

        self.refPointDiffVal = \
            spec.chebDiffMatrix(self.approxOrder, a=-1, b=1).\
                dot(self.refPointVal)
        match elemType:
            case 0:
                self.elemType = 0
                a = self.interval[0]; b = self.interval[1]
                q = (b - a) / 2.0; p = (b + a) / 2.0
                self.map = lambda x: (q * x + p)
                self.inverseMap = lambda x: (x - p)/q

                self.derivativeMap = lambda x: 1.0 / (q + x * 0)
                self.inverseDerivativeMap = lambda x: q + 0
            case 1:
                self.elemType = 1
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
                self.elemType = 2
                # if self.interval[0] == -np.inf:
                #     self.map = lambda x: -((1.0 - x) / (1.0 + x) - self.interval[1])
                #     self.inverseMap = lambda x: (x + 1.0 - self.interval[1])/(-x + self.interval[1] + 1.0)
                #
                #     self.derivativeMap = lambda x: (x + 1) ** 2 / 2
                #     self.inverseDerivativeMap = lambda x: 2/(x + 1)**2
                #     return

                if self.interval[1] == np.inf:
                    self.map = lambda x: np.log((1.0 + x) / (1.0 - x) + 1) + self.interval[0]
                    self.inverseMap = lambda x: np.exp(self.interval[0] - x)*(np.exp(x - self.interval[0]) - 2.0)

                    self.derivativeMap = lambda x: (1.0 - x)
                    self.inverseDerivativeMap = lambda x: 1/(1.0 - x)
                    return
            case 3:
                self.elemType = 3
                a = 0;
                b = np.pi*2
                q = (b - a) / 2.0;
                p = (b + a) / 2.0
                self.map = lambda x: (q * x + p)
                self.inverseMap = lambda x: (x - p) / q

                self.derivativeMap = lambda x: 1.0 / (q + x * 0)
                self.inverseDerivativeMap = lambda x: q + 0
            case 4:
                self.elemType = 4



    def evalAtChebPoints(self):
        return self.refPointVal
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
        if self.elemType <= 2:
            x = np.atleast_1d(x)
            basisMatrix = self.refPointVal
            if x.size == 1:
                if x[0] == self.interval[0]:
                    return np.array([basisMatrix[0, :]])
                if x[0] == self.interval[-1]:
                    return np.array([basisMatrix[-1, :]])

            result = spec.barycentricChebInterpolate(basisMatrix, x, a=-1, b=1, axis=0)
        elif self.elemType == 3:
            x = self.map(np.atleast_1d(x))
            n = self.approxOrder

            xj = 2*np.pi*np.arange(n)/n

            result = 1.0/n*np.sin(n*(x[:, np.newaxis]-xj[np.newaxis, :])/2.0)/np.tan((x[:, np.newaxis]-xj[np.newaxis, :])/2.0)
        return result
    def getMappedRefPoints(self):
        if self.elemType < 3:
            x = spec.chebNodes(self.approxOrder, -1, 1)
            x = self.map(x)
            return x
        elif self.elemType == 3:
            return np.arange(self.approxOrder)*2*np.pi/self.approxOrder
        elif self.elemType == 4:
            return np.arange(self.approxOrder)*np.pi/self.approxOrder

    def evalDiff(self, x):
        """ Evaluates derivatives of basis functions at points x

            Arguments:
                x: evaluation points

            Returns:
                result: array with the shape: (*x.shape, degree of element)
        """
        if self.elemType <= 2:
            x = np.atleast_1d(x)
            derivativeBasisMatrix = self.refPointDiffVal
            if x.size == 1:
                if x[0] == self.interval[0]:
                    return self.derivativeMap(-1)*np.array([derivativeBasisMatrix[0, :]])
                if x[0] == self.interval[-1]:
                    return self.derivativeMap(1)*np.array([derivativeBasisMatrix[-1, :]])
            # xs = np.array(self.imap(x), dtype=np.float)
            result = spec.barycentricChebInterpolate(f=derivativeBasisMatrix, x=x, a=-1, b=1, axis=0) \
                     * np.reshape(self.derivativeMap(x), (*x.shape, 1))
        elif self.elemType == 3:
            x = self.map(np.atleast_1d(x))
            n = self.approxOrder
            xj = 2 * np.pi * np.arange(n) / n
            result = -(-1)**(np.arange(n)) / (2.0 * n) * ((n * np.cos(n * (x[:, np.newaxis]) / 2.0) \
                                / np.tan((-x[:, np.newaxis] + xj[np.newaxis, :]) / 2.0)) +
                                np.sin(n * x[:, np.newaxis] / 2.0)\
                                / (np.sin((xj[np.newaxis, :] - x[:, np.newaxis])/2.0)**2))
        return result
    def generateFunction(self):
        return lambda x: self.eval(x)
    def generateDerivativeFunction(self):
        return lambda x: self.evalDiff(x)