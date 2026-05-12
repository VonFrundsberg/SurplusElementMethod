import numpy as np
from matplotlib import legend

from SurplusElement.Basic_Galerkin import Galerkin1d as galerkin
import SurplusElement.Basic_Galerkin.element.Element1d.element1dUtils as elem1dUtils
import SurplusElement.Basic_Galerkin.Mesh.mesh as MeshClass

from SurplusElement.mathematics import integrate as integr
import matplotlib.pyplot as plt


"""
The problem is 
-d2(u/dx2) - d(u/dx) = exp(-x) on the [0, inf] interval with dirichlet BC set to u(0) = 0.0, u(inf) = 0.0
"""
def fun(approximationOrder, amountOfElements, integrationPointsAmount = 500):
    galerkinMethodObject = galerkin.GalerkinMethod1d("LS")
    domainSize = np.inf
    infElementBoundary = 10.0

    fluxForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm2(
        trialElement, testElement, lambda x: x * 0.0 + 1.0, integrationPointsAmount)
    gradForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm1(
        trialElement, testElement, lambda x: x * 0.0 + 1.0, integrationPointsAmount)

    def fluxDGForm(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_centralFlux(
            trialElement=trialElement, testElement=testElement, weight=lambda x: 0 * x - 1.0,
            physicalBoundary=np.array([0.0, domainSize]))

    def fluxBoundaryForm(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateBilinearFormAtBoundary1(
            trialElement=trialElement, testElement=testElement, weight=lambda x: 0.0 * x - 1.0,
            B=domainSize)

    eps = 1e-15
    sigma = 1 * approximationOrder ** 2 / (domainSize / amountOfElements)
    def DGForm1(trialElement: galerkin.element.Element1d, elementTest: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_JumpComponentMain(
            trialElement=trialElement, testElement=elementTest, weight=lambda x: x * 0.0 - 1.0,
            physicalBoundary=np.array([0, domainSize]), eps=eps)

    def DGForm2(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_JumpComponentSymmetry(
            trialElement=trialElement, testElement=testElement, weight=lambda x: x * 0.0 - 1.0,
            physicalBoundary=np.array([0, domainSize]), eps=eps)

    def DGForm3(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_ErrorComponent(
            trialElement=trialElement, testElement=testElement, weight=lambda x: x * 0.0 + sigma,
            physicalBoundary=np.array([0, domainSize]), eps=eps)


    functional = lambda testElement: elem1dUtils.integrateFunctional(
        testElement=testElement, function=lambda x: np.exp(-x),
            weight=lambda x: x * 0.0 + 1.0, integrationPointsAmount=integrationPointsAmount)


    galerkinMethodObject.setBilinearForm(innerForms=[gradForm, fluxForm],
                                         discontinuityForms=[DGForm1, DGForm2, DGForm3, fluxDGForm],
                                         boundaryForms=[fluxBoundaryForm]
                                         )

    galerkinMethodObject.setRHSFunctional([functional])
    boundaryConditions = ['{"boundaryPoint": "0.0", "boundaryValue": 0.0}',
                          '{"boundaryPoint": "np.inf", "boundaryValue": 0.0}']
    # boundaryConditions = []
    galerkinMethodObject.setDirichletBoundaryConditions(boundaryConditions)

    mesh = MeshClass.mesh(1)
    mesh.generateUniformMeshOnRectange([0.0, infElementBoundary],
                                       [amountOfElements],
                                       [approximationOrder])
    mesh.extendRectangleToInf_AlongAxis_OneDirection("right", infElementBoundary, 0, approximationOrder)

    mesh.establishNeighbours()

    mesh.fileWrite("elementsData.txt", "neighboursData.txt")
    mesh.fileRead("elementsData.txt", "neighboursData.txt")
    galerkinMethodObject.initializeMesh(mesh)
    galerkinMethodObject.initializeElements()
    galerkinMethodObject.calculateElements()
    galerkinMethodObject.solveSLAE()

    # galerkinMethodObject.checkPositiveEigenvalues()


    w, grid = integr.reg_22_wn(0.0, 10.0, integrationPointsAmount)

    gridSolution = galerkinMethodObject.evaluateSolution(grid)
    """Zero Dirichlet BC"""
    analyticSolution = lambda x: x * np.exp(-x)
    # print(gridSolution[0], gridSolution[-1])
    # print(galerkinMethodObject.solution[0] - 10.0,
    #       galerkinMethodObject.solution[-1] - 1.0)

    # plt.plot(grid, gridSolution - analyticSolution(grid), label="approximation")
    # print(approximationOrder, np.max(np.abs(gridSolution - analyticSolution(grid))))
    print(approximationOrder, np.max(np.abs(gridSolution - analyticSolution(grid))))
    # plt.show()
    # plt.plot(grid, gridSolution, label="approximation")
    # plt.plot(grid, analyticSolution(grid), label="exact solution")
    # plt.legend()
    # plt.show()

import time as time
for i in range(2, 101):
    fun(i, 2, integrationPointsAmount=10000)
    # time.sleep(5)