class DirichletBoundaryCondition:
    boundaryPoint: float
    boundaryValue: float

    def __str__(self):
        return str(self.boundaryPoint) + " " + str(self.boundaryValue)

class ApproximationSpaceParameter:
    """
    shift is used in Laguerre functions
    s is used in Exponential and Rational (0, inf) mapped polynomials
    """
    shift: float
    s: float
    def __init__(self):
        self.shift = 0.0
        self.s = 1.0
    def __str(self):
        return ("shift: " + str(self.shift) +
                ", s: " + str(self.s))