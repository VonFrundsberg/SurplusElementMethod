import numpy as np

class PeriodicCardinalElement:
    def __init__(self, interval, approxOrder, dirichletBoundaryConditions=None):
        a = 0;
        b = np.pi * 2
        q = (b - a) / 2.0;
        p = (b + a) / 2.0
        self.map = lambda x: (q * x + p)
        self.inverseMap = lambda x: (x - p) / q

        self.derivativeMap = lambda x: 1.0 / (q + x * 0)
        self.inverseDerivativeMap = lambda x: q + 0

    def eval(self, x):
        """ Evaluates basis functions at points x

                    Arguments:
                        x: evaluation points

                    Returns:
                        result: array with the shape: (*x.shape, degree of element)
                """
        x = self.map(np.atleast_1d(x))
        n = self.approxOrder

        xj = 2 * np.pi * np.arange(n) / n

        result = 1.0 / n * np.sin(n * (x[:, np.newaxis] - xj[np.newaxis, :]) / 2.0) / np.tan(
            (x[:, np.newaxis] - xj[np.newaxis, :]) / 2.0)

        return result

    def evalDiff(self, x):
        """ Evaluates derivatives of basis functions at points x

            Arguments:
                x: one-dimensional array of evaluation points

            Returns:
                result: array with the shape:  (approximation order of element, len(x))
        """
        x = self.map(np.atleast_1d(x))
        n = self.approxOrder
        xj = 2 * np.pi * np.arange(n) / n
        result = -(-1)**(np.arange(n)) / (2.0 * n) * ((n * np.cos(n * (x[:, np.newaxis]) / 2.0) \
                                / np.tan((-x[:, np.newaxis] + xj[np.newaxis, :]) / 2.0)) +
                                np.sin(n * x[:, np.newaxis] / 2.0)\
                                / (np.sin((xj[np.newaxis, :] - x[:, np.newaxis])/2.0)**2))
        return result