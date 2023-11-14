import numpy as np

class ExponentialInfHalfSpaceElement:
    def __init__(self, interval, approxOrder, dirichletBoundaryConditions=None):
        self.interval = np.array(interval)
        self.approxOrder = int(approxOrder)
        self.dirichletBoundaryConditions = dirichletBoundaryConditions
        if self.interval[1] == np.inf:
            self.map = lambda x: np.log((1.0 + x) / (1.0 - x) + 1) + self.interval[0]
            self.inverseMap = lambda x: np.exp(self.interval[0] - x) * (np.exp(x - self.interval[0]) - 2.0)

            self.derivativeMap = lambda x: (1.0 - x)
            self.inverseDerivativeMap = lambda x: 1 / (1.0 - x)