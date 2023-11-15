class DirichletBoundaryCondition:
    boundaryPoint: float
    boundaryValue: float

    def __str__(self):
        return str(self.boundaryPoint) + " " + str(self.boundaryValue)