import numpy as np
from matplotlib import legend

from SurplusElement.GalerkinMethod import Galerkin1d as galerkin
import SurplusElement.GalerkinMethod.element.Element1d.element1dUtils as elem1dUtils
import SurplusElement.GalerkinMethod.Mesh.mesh as MeshClass

from SurplusElement.mathematics import integrate as integr
import matplotlib.pyplot as plt


"""

The problem is 
-d2u/dx^2 + u(x) = exp(-x) on the [0, inf] interval with dirichlet BC set to u(0) = 9, u(inf) = 0
"""
def fun(approximationOrder, amountOfElements, integrationPointsAmount = 500):
    domainSize = np.inf
    galerkinMethodObject = galerkin.GalerkinMethod1d("LS")

    gradForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm1(
        trialElement, testElement, lambda x: np.nan_to_num(x=x * 0.0, nan=0.0) + 1.0, integrationPointsAmount)

    innerForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm0(
        trialElement, testElement, lambda x: np.nan_to_num(x=x * 0.0, nan=0.0) + 1.0, integrationPointsAmount)
    sigma = 1e+5*approximationOrder ** 2 / (10.0 / amountOfElements)

    def minusSignFunction(x):
        if x == 0.0:
            return np.nan_to_num(x=x * 0.0, nan=0.0) - 1.0
        if x == domainSize:
            return np.nan_to_num(x=x * 0.0, nan=0.0) + 0.0
        return np.nan_to_num(x=x * 0.0, nan=0.0)

    def boundaryForm1(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateBilinearFormAtBoundary_21(
            trialElement=trialElement, testElement=testElement, weight=lambda x: -minusSignFunction(x))

    def boundaryForm2(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateBilinearFormAtBoundary_20(
            trialElement=trialElement, testElement=testElement, weight=lambda x: -minusSignFunction(x))

    def boundaryForm3(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateBilinearFormAtBoundary1(
            trialElement=trialElement, testElement=testElement,
            weight=lambda x: np.nan_to_num(x=x * 0.0, nan=0.0) + sigma, B=0.0)

    functional = lambda testElement: elem1dUtils.integrateFunctional(
        testElement=testElement, function=lambda x: (np.exp(-x)),
            weight=lambda x: np.nan_to_num(x=x * 0.0, nan=0.0) + 1.0, integrationPointsAmount=integrationPointsAmount)


    boundaryFunctional0 = lambda testElement: elem1dUtils.evaluateFunctionalAtBoundaries0(
        testElement=testElement, weight=lambda x: minusSignFunction(x),
        leftValue=-9.0, rightValue=-0.0, leftBoundary=0.0, rightBoundary=domainSize)
    def BCfunction(x):
        if x == 0.0:
            return np.nan_to_num(x=x * 0.0, nan=0.0) + 9.0
        if x == domainSize:
            return np.nan_to_num(x=x * 0.0, nan=0.0) + 0.0
        return np.nan_to_num(x=x * 0.0, nan=0.0)
    boundaryFunctional1 = lambda testElement: elem1dUtils.evaluateFunctionalAtBoundaries1(
        testElement=testElement, weight=lambda x: BCfunction(x) * sigma,
        leftBoundary=0.0, rightBoundary=domainSize)

    galerkinMethodObject.setBilinearForm(innerForms=[gradForm, innerForm],
                                         discontinuityForms=[],
                                         boundaryForms=[boundaryForm1, boundaryForm2,
                                                        boundaryForm3]
                                         )
    galerkinMethodObject.setRHSFunctional([functional, boundaryFunctional0, boundaryFunctional1])
    # galerkinMethodObject.setRHSFunctional([functional])
    boundaryConditions = ['{"boundaryPoint": "0.0", "boundaryValue": 1.0}',
                          '{"boundaryPoint": "np.inf", "boundaryValue": 0.0}']
    # boundaryConditions = []
    galerkinMethodObject.setDirichletBoundaryConditions(boundaryConditions)

    mesh = MeshClass.mesh(1)
    # infElementBoundary = 1.0
    # mesh.generateUniformMeshOnRectange([0, infElementBoundary],
    #                                    [amountOfElements],
    #                                    [approximationOrder])
    # mesh.extendRectangleToInf_AlongAxis_OneDirection("right", infElementBoundary, 0, approximationOrder)
    fileElements = open("elementsData.txt", "w")
    fileElements.write("0.0 inf " + str(int(approximationOrder)) + " 1.0" + "\n")
    fileNeighbours = open("neighboursData.txt", "w")
    fileElements.close()
    fileNeighbours.close()
    # mesh.extendRectangleToInf_AlongAxis_OneDirection("right", 0.0, 0, approximationOrder)
    # mesh.establishNeighbours()

    # mesh.fileWrite("elementsData.txt", "neighboursData.txt")
    mesh.fileRead("elementsData.txt", "neighboursData.txt")
    galerkinMethodObject.initializeMesh(mesh)
    galerkinMethodObject.initializeElements()
    galerkinMethodObject.calculateElements()
    galerkinMethodObject.solveSLAE()


    w, grid = integr.reg_22_wn(8.0, 10.0, integrationPointsAmount)

    gridSolution = galerkinMethodObject.evaluateSolution(grid)
    analyticSolution = lambda x: 0.5 * (np.exp(-x) * (18.0 + x))
    analyticSolutionD2 = lambda x: 0.5 * (np.exp(-x) * (16.0 + x))
    gridSol = galerkinMethodObject.evaluateSolution(grid)
    d1gridSol = galerkinMethodObject.evaluateSolutionDerivative(grid)
    d2gridSol = galerkinMethodObject.evaluateSolutionDerivative(grid, 2)

    # print(approximationOrder, np.max(np.abs(gridSolution - analyticSolution(grid))))
    # plt.plot(grid, g_V(grid), label="rhs")
    # plt.show()
    # print(i, galerkinMethodObject.solution[0] - 9.0, galerkinMethodObject.solution[-1] - 0.0)
    # print(i, np.abs(galerkinMethodObject.solution[0] - 9.0))
    # print(i, np.max(np.abs(-d2gridSol + gridSol - np.exp(-grid))))
    # plt.plot(grid,d2gridSol)
    # plt.plot(grid, analyticSolutionD2(grid))
    # print(i,
    #       np.max(np.abs(gridSol - analyticSolution(grid)))/np.max(np.abs(analyticSolution(grid))),
    #       np.max(np.abs(d2gridSol - analyticSolutionD2(grid)))/np.max(np.abs(analyticSolutionD2(grid))))
    print(i, np.sqrt(w @ (gridSol - analyticSolution(grid))**2) / np.sqrt(w @ (analyticSolution(grid))**2),
          np.sqrt(w @ (d2gridSol - analyticSolutionD2(grid))**2) / np.sqrt(w @ (analyticSolutionD2(grid))**2),
          np.sqrt(w @ (-d2gridSol + gridSol - np.exp(-grid))**2) / np.sqrt(w @ (np.exp(-grid))**2))
    # plt.plot(grid, gridSolution, label="approximation")
    # plt.plot(grid, analyticSolution(grid), label="exact solution")
    # plt.legend()
    # plt.show()
    # plt.plot(grid, -d2gridSol + gridSol, label="numerical LHS")
    # plt.plot(grid, np.exp(-grid), label="exact RHS")
    # plt.show()

for i in range(3, 50):
    fun(i, 1, integrationPointsAmount=100000)