import numpy as np
import GalerkinMethod.mesh.mesh as MeshClass
import GalerkinMethod.Galerkin1d as galerkin
import GalerkinMethod.element.Element1d.element1dUtils as elem1dUtils
import time as time
from mathematics import integrate as integr
import matplotlib.pyplot as plt
import scipy.optimize as sp_opt
import mathematics.spectral as spec

def fun(meshArg, approximationOrder, mappingType, integrationPointsAmount = 500):
    galerkinMethodObject = galerkin.GalerkinMethod1d()

    gradForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm1(
        trialElement, testElement, lambda x: x * x, integrationPointsAmount)

    def boundaryForm1(trialElement: galerkin.element.Element1d, elementTest: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_JumpComponentMain(
            trialElement=trialElement, testElement=elementTest, weight=lambda x: -x * x)

    def boundaryForm2(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_JumpComponentSymmetry(
            trialElement=trialElement, testElement=testElement, weight=lambda x: -x * x)

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

    boundaryConditions = ['{"boundaryPoint": "np.inf", "boundaryValue": 0.0}']

    galerkinMethodObject.setDirichletBoundaryConditions(boundaryConditions)

    mesh = MeshClass.mesh(1)
    mesh.generateProvidedMeshOnRectange([meshArg],
                                       [approximationOrder], [mappingType])
    mesh.establishNeighbours()

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


    w, grid = integr.log_16_wn(0.0, mesh.elements[0][0][1], integrationPointsAmount)

    gridSolution = galerkinMethodObject.evaluateSolutionAtPoints(grid)
    a = mesh.elements[0][0][1]
    error = np.sum(w*(2*np.log(grid)*(gridSolution - 2.0) + (gridSolution - 2.0)**2)) + a * (2.0 + (-2.0 + np.log(a))*np.log(a))

    for i in range(1, mesh.elementsAmount - 1):
        w, grid = integr.reg_22_wn(mesh.elements[i][0][0], mesh.elements[i][0][1], integrationPointsAmount)
        calculatedDimensionless_BPL_asol = dimensionless_BPL_asol(grid, 2.0, 3.0)
        gridSolution = galerkinMethodObject.evaluateSolutionAtPoints(grid)
        error += np.sum(w * (calculatedDimensionless_BPL_asol - gridSolution) ** 2)

    w, grid = integr.reg_32_wn(-1, 1, integrationPointsAmount)

    mappedGrid = galerkinMethodObject.elements[-1].map(grid)
    calculatedDimensionless_BPL_asol = dimensionless_BPL_asol(mappedGrid, 2.0, 3.0)
    gridSolution = galerkinMethodObject.evaluateSolutionAtPoints(mappedGrid)
    # plt.plot(grid, calculatedDimensionless_BPL_asol)
    # plt.plot(grid, gridSolution)
    # plt.show()
    error += np.sum(w * galerkinMethodObject.elements[-1].derivativeMap(grid) * (calculatedDimensionless_BPL_asol - gridSolution) ** 2)
    # plt.plot(grid, w * galerkinMethodObject.elements[-1].derivativeMap(grid) * (calculatedDimensionless_BPL_asol - gridSolution) ** 2)
    # plt.show()
    # plt.plot(grid, gridSolution - calculatedDimensionless_BPL_asol)
    # plt.show()
    return error


indices = []
errors = []
for i in range(4, 100, 4):
    indices.append(i)
    bounds =[(0.0, 0.5), (0.5, 1.0), (1.0, 5.0),
             (0.0, 1.0),  (0.0, 1.0),  (0.0, 1.0), (0.0, 1.0)]
    def lambdaFun(x):
        approxOrdersAmount = i*4
        totalApproxWeight = np.sum(x[3:])
        approxOrdersFloat = approxOrdersAmount*np.array([x[3:]/totalApproxWeight], dtype=float)
        approxOrders = np.array(approxOrdersFloat, dtype=int) + 4*np.ones([4], dtype=int)
        mesh = [[0.0, x[0], x[1], x[2], np.inf]]
        result = fun(mesh, np.squeeze(approxOrders), [0, 0, 0, 1], 2000)
        return result

    error = sp_opt.direct(lambdaFun, bounds, maxiter=20)

    x = error.get('x')
    approxOrdersAmount = i * 4
    totalApproxWeight = np.sum(x[3:])
    approxOrdersFloat = approxOrdersAmount * np.array([x[3:] / totalApproxWeight], dtype=float)
    approxOrders = np.array(approxOrdersFloat, dtype=int) + 2 * np.ones([4], dtype=int)
    mesh = [[0.0, x[0], x[1], x[2], np.inf]]
    result = fun(mesh, np.squeeze(approxOrders), [0, 0, 0, 1], 2000)
    print(i)
    print(np.squeeze(mesh))
    print(np.squeeze(approxOrders))
    print(result)
    errors.append(error)

plt.loglog(indices, errors)
plt.show()