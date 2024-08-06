import numpy as np
from SurplusElement import GalerkinMethod as MeshClass, GalerkinMethod as galerkin, GalerkinMethod as elem1dUtils
from SurplusElement.mathematics import integrate as integr
import matplotlib.pyplot as plt


def fun(approximationOrder, infElementBoundary, amountOfElementsOnFiniteGrid, integrationPointsAmount = 500):
    galerkinMethodObject = galerkin.GalerkinMethod1d()

    gradForm = "integral w(x) grad(u) @ grad(v)"
    boundaryForm1 = "boundaryIntegral w(x) [u] <grad(v) @ n>"
    boundaryForm2 = "boundaryIntegral w(x) [v] <grad(u) @ n>"

    gradForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm1(
        trialElement, testElement, lambda x: x * x, integrationPointsAmount)

    def boundaryForm1(trialElement: galerkin.element.Element1d, elementTest: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_JumpComponentMain(
            trialElement=trialElement, testElement=elementTest, weight=lambda x: -x * x)

    def boundaryForm2(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_JumpComponentSymmetry(
            trialElement=trialElement, testElement=testElement, weight=lambda x: -x * x)

    functional = "integral w(x) u f"
    def dimensionless_BPL_function(x, gamma: float, beta: float):
        lessThanOneArgs = np.where(x <= 1)
        moreThanOneArgs = np.where(x >= 1)
        result = np.zeros(x.shape)
        result[lessThanOneArgs] = x[lessThanOneArgs]**(-gamma)
        result[moreThanOneArgs] = x[moreThanOneArgs]**(-beta)
        # plt.plot(x, result)
        # plt.show()
        return result
    fromJacobian = 2
    gamma = 2 - fromJacobian
    beta = 3 - fromJacobian
    functional = lambda testElement: elem1dUtils.integrateFunctional(
        testElement=testElement, function=lambda x: dimensionless_BPL_function(x, gamma, beta), weight=lambda x: 1, integrationPointsAmount=integrationPointsAmount)

    galerkinMethodObject.setBilinearForm(innerForms=[gradForm], boundaryForms=[boundaryForm1, boundaryForm2])
    galerkinMethodObject.setRHSFunctional([functional])
    # boundaryConditions = []
    # boundaryConditions = ['{"boundaryPoint": "np.pi", "boundaryValue": 0.0}',
    #                       '{"boundaryPoint": "0", "boundaryValue": 0.0}']

    boundaryConditions = ['{"boundaryPoint": "np.inf", "boundaryValue": 0.0}']

    galerkinMethodObject.setDirichletBoundaryConditions(boundaryConditions)

    mesh = MeshClass.mesh(1)
    mesh.generateSkewedMeshOnRectange([0, infElementBoundary],
                                       [amountOfElementsOnFiniteGrid],
                                       [approximationOrder])

    mesh.extendRectangleToInf_AlongAxis_OneDirection("right", infElementBoundary, 0, approximationOrder*2)
    mesh.establishNeighbours()

    # np.set_printoptions(precision=3, suppress=True)

    mesh.fileWrite("elementsData.txt", "neighboursData.txt")
    mesh.fileRead("elementsData.txt", "neighboursData.txt")
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
    calculatedDimensionless_BPL_asol = dimensionless_BPL_asol(grid, 2.0, 3.0)
    # errorFrom1toInf = np.sum(w * (2 * calculatedDimensionless_BPL_asol - gridSolution) ** 2)
    # plt.plot(grid, gridSolution - calculatedDimensionless_BPL_asol)
    # plt.show()
    return errorFrom0to1


indices = []
errors = []
for i in range(4, 100, 1):
    indices.append(i)
    error = fun(i, 4.0, 6, 2000)
    print(i, error)
    errors.append(error)

plt.loglog(indices, errors)
plt.show()