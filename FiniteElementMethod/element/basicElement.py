import numpy as np
from misc import spectral as sp

class basicElem():
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
        # if Id == None:
        #     self.Id = np.eye(n)
        # else:
        #     self.Id = Id
        # self.D = sp.chebDiff(self.polynomialOrder).dot(self.rp_eval())
        self.functionAtReferencePoints = np.eye(polynomialOrder)
        for it in self.boundaryConditions:
            self.functionAtReferencePoints[it[0], it[0]] = it[1]
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

    def evaluateChebyshevPoints(self):
        return self.functionAtChebyshevPoints

    def evaluatePoints(self, x):
        """Constructor of basic (one-dimensional) element

                Arguments:
                    interval:
                    polynomialOrder:
                    mappingType: 0 is linear qx+p, 1 is rational (1+x)/(1-x), 2 is exponential ???
                    boundaryConditions:

                Returns: return shape: (*x.shape, deg of element)
                """
        x = np.atleast_1d(x)
        P = self.rp_eval()
        if x.size == 1:
            if x[0] == self.I[0]:
                return np.array([P[0, :]])
            if x[0] == self.I[-1]:
                return np.array([P[-1, :]])
            # print('oops')
        # xs = np.array(self.imap(x), dtype=np.float)
        res = sp.bary(P, x)
        return res

    def xs(self):
        x = sp.chebNodes(self.n)
        x = self.map(x)
        return x

    def new_xs(self):
        x = sp.chebNodes(self.n)
        mx = self.imap

        return x, mx

    def evaluateDerivativePoints(self, x): ## return shape: (*x.shape, deg of element)
        x = np.atleast_1d(x)
        P = self.D
        if x.size == 1:
            if x[0] == self.interval[0]:
                return self.dmap(-1)*np.array([P[0, :]])
            if x[0] == self.interval[-1]:
                return self.dmap(1)*np.array([P[-1, :]])
        # xs = np.array(self.imap(x), dtype=np.float)
        res = sp.bary(f=P, x=x)*np.reshape(self.dmap(x), (*x.shape, 1))
        return res
    def gen_func(self):
        return lambda x: self.p_eval(x)
    def gen_funcd(self):
        return lambda x: self.dp_eval(x)

    def __call__(self, x):
            return self.p_eval(x)
    def d(self, x):
            return self.dp_eval(x)