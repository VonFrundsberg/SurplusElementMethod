import numpy as np
import GalerkinMethod.mesh.mesh as MeshClass
import GalerkinMethod.Galerkin1d as galerkin
import GalerkinMethod.element.Element1d.element1dUtils as elem1dUtils
import time as time
from mathematics import integrate as integr
import matplotlib.pyplot as plt
import mathematics.spectral as spec

def fun(approximationOrder,  integrationPointsAmount = 500):
    galerkinMethodObject = galerkin.GalerkinMethod1d()

    gradForm = "integral w(x) grad(u) @ grad(v)"

    gradForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm1(
        trialElement, testElement, lambda x: x * x, integrationPointsAmount)

    functional = "integral w(x) u f"
    def dimensionless_BPL_function(x, gamma: float, beta: float):
        lessThanOneArgs = np.where(x <= 1)
        moreThanOneArgs = np.where(x >= 1)
        result = np.zeros(x.shape)
        result[lessThanOneArgs] = x[lessThanOneArgs]**(-gamma)
        result[moreThanOneArgs] = x[moreThanOneArgs]**(-beta)
        return result

    fromJacobian = 2
    gamma = 2 - fromJacobian
    beta = 3 - fromJacobian
    functional = lambda testElement: elem1dUtils.integrateFunctional(
        testElement=testElement, function=lambda x: dimensionless_BPL_function(x, gamma, beta), weight=lambda x: 1, integrationPointsAmount=integrationPointsAmount)

    galerkinMethodObject.setBilinearForm(innerForms=[gradForm], boundaryForms=[])
    galerkinMethodObject.setRHSFunctional([functional])
    # boundaryConditions = []
    # boundaryConditions = ['{"boundaryPoint": "np.pi", "boundaryValue": 0.0}',
    #                       '{"boundaryPoint": "0", "boundaryValue": 0.0}']

    boundaryConditions = ['{"boundaryPoint": "np.inf", "boundaryValue": 0.0}']

    galerkinMethodObject.setDirichletBoundaryConditions(boundaryConditions)

    mesh = MeshClass.mesh(1)
    f = open("elementsDataSpectral.txt", "w")
    f.write("0.0 inf " + str(approximationOrder) + " 1.0")
    f.close()
    mesh.fileRead("elementsDataSpectral.txt", "neighboursDataSpectral.txt")
    galerkinMethodObject.initializeMesh(mesh)
    galerkinMethodObject.initializeElements()
    galerkinMethodObject.calculateElements()
    galerkinMethodObject.solveSLAE()



    def dimensionless_BPL_asol(x, gamma: float, beta: float):
        result = np.zeros(x.shape)
        lessThanOneArgs = np.where(x <= 1)
        moreThanOneArgs = np.where(x >= 1)

        if gamma == 2.0:
            result[lessThanOneArgs] = (beta - 1.0)/(beta - 2.0) - np.log(x[lessThanOneArgs])
        if beta == 3.0:
            result[moreThanOneArgs] = 1/(x[moreThanOneArgs]) * ((4.0 - gamma)/(3.0 - gamma) + np.log(x[moreThanOneArgs]))
        return result


    w, grid = integr.log_16_wn(0.0, 1.0, integrationPointsAmount)

    gridSolution = galerkinMethodObject.evaluateSolutionAtPoints(grid)

    errorFrom0to1 = np.sum(w*(2*np.log(grid)*(gridSolution - 2.0) + (gridSolution - 2.0)**2)) + 2.0

    w, grid = integr.reg_22_wn(-1.0, 1.0, integrationPointsAmount)
    map = lambda x: (1 + x)/(1 - x) + 1
    calculatedDimensionless_BPL_asol = dimensionless_BPL_asol(map(grid), 2.0, 3.0)
    gridSolution = galerkinMethodObject.evaluateSolutionAtPoints(map(grid))
    errorFrom1toInf = np.sum(w * (calculatedDimensionless_BPL_asol - gridSolution) ** 2)
    # plt.plot(grid, gridSolution - calculatedDimensionless_BPL_asol)
    # plt.show()
    return galerkinMethodObject.getAmountOfNonZeroSLAE_elements(), errorFrom0to1 + errorFrom1toInf


indices = []
errors = []
for i in range(4, 100, 1):
    indices.append(i)
    nonzero, error = fun(i, 2000)
    print(nonzero, error)
    errors.append(error)

plt.loglog(indices, errors)
plt.show()