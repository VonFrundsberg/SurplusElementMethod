import numpy as np
from matplotlib import legend

from SurplusElement.GalerkinMethod import Galerkin1d as galerkin
import SurplusElement.GalerkinMethod.element.Element1d.element1dUtils as elem1dUtils
import SurplusElement.GalerkinMethod.Mesh.mesh as MeshClass

from SurplusElement.mathematics import integrate as integr
import matplotlib.pyplot as plt


"""

The problem is 
-d2u/dx^2 + du/dx = exp(-x) + 1 on the [0, 10] interval with dirichlet BC set to u(0) = 10, u(10) = 1
Treatment of du/dx is through the upwinding scheme
Mesh points and approximation orders are uniform and 
being controlled by amountOfElements and approximationOrder respectively
"""
def fun(approximationOrder, amountOfElements, integrationPointsAmount = 500):
    domainSize = 10.0
    galerkinMethodObject = galerkin.GalerkinMethod1d("LS")

    gradForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm1(
        trialElement, testElement, lambda x: x * 0.0 + 1.0, integrationPointsAmount)

    fluxForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm2(
        trialElement, testElement, lambda x: x * 0.0 + 1.0, integrationPointsAmount)
    eps = 1e-14

    sigma = approximationOrder ** 2 / (10.0 / amountOfElements)
    def DGForm1(trialElement: galerkin.element.Element1d, elementTest: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_JumpComponentMain(
            trialElement=trialElement, testElement=elementTest, weight=lambda x: x * 0.0 - 1.0,
            physicalBoundary=np.array([0, domainSize]), eps = eps)
    def DGForm2(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_JumpComponentSymmetry(
            trialElement=trialElement, testElement=testElement, weight=lambda x: x * 0.0 - 1.0,
            physicalBoundary=np.array([0, domainSize]), eps = eps)
    def DGForm3(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_ErrorComponent(
            trialElement=trialElement, testElement=testElement, weight=lambda x: x * 0.0 + sigma,
            physicalBoundary=np.array([0, domainSize]), eps = eps)
    def fluxDGForm(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_upwind(
            trialElement=trialElement, testElement=testElement, weight=lambda x: x * 0.0 - 1.0,
            physicalBoundary=np.array([0, domainSize]), eps = eps)


    def minusSignFunction(x):
        if x == 0:
            return x * 0 - 1.0
        if x == domainSize:
            return x * 0 + 1.0
        return x * 0

    def boundaryForm1(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateBilinearFormAtBoundary_21(
            trialElement=trialElement, testElement=testElement, weight=lambda x: -minusSignFunction(x))

    def boundaryForm2(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateBilinearFormAtBoundary_20(
            trialElement=trialElement, testElement=testElement, weight=lambda x: -minusSignFunction(x))

    def boundaryForm3(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateBilinearFormAtBoundary1(
            trialElement=trialElement, testElement=testElement, weight=lambda x: x * 0.0 + sigma, B=0.0)

    def boundaryForm4(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateBilinearFormAtBoundary1(
            trialElement=trialElement, testElement=testElement, weight=lambda x: x * 0.0 + sigma, B=domainSize)

    def boundaryForm5(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateBilinearFormAtBoundary1(
            trialElement=trialElement, testElement=testElement, weight=lambda x: x * 0.0 - 1.0, B=domainSize)

    functional = lambda testElement: elem1dUtils.integrateFunctional(
        testElement=testElement, function=lambda x: (np.exp(-x) + 1.0),
            weight=lambda x: x * 0.0 + 1.0, integrationPointsAmount=integrationPointsAmount)

    boundaryFunctional0 = lambda testElement: elem1dUtils.evaluateFunctionalAtBoundaries0(
        testElement=testElement, weight=lambda x: minusSignFunction(x),
        leftValue=-10.0, rightValue=-1.0, leftBoundary=0.0, rightBoundary=domainSize)
    def BCfunction(x):
        if x == 0:
            return x * 0.0 + 10.0
        if x == domainSize:
            return x * 0.0 + 1.0
        return x * 0.0
    boundaryFunctional1 = lambda testElement: elem1dUtils.evaluateFunctionalAtBoundaries1(
        testElement=testElement, weight=lambda x: BCfunction(x) * sigma,
        leftBoundary=0.0, rightBoundary=domainSize)

    galerkinMethodObject.setBilinearForm(innerForms=[gradForm, fluxForm],
                                         discontinuityForms=[DGForm1, DGForm2, fluxDGForm, DGForm3],
                                         boundaryForms=[boundaryForm1, boundaryForm2,
                                                        boundaryForm3, boundaryForm4, boundaryForm5]
                                         )

    def ZeroLeftFunction(x):
        if x == 0:
            return x * 0.0 + 0.0
        if x == domainSize:
            return x * 0.0 + 1.0
        return x * 0.0
    boundaryFunctional2 = lambda testElement: elem1dUtils.evaluateFunctionalAtBoundaries1(
        testElement=testElement, weight=lambda x: 10.0*ZeroLeftFunction(x),
        leftBoundary=0.0, rightBoundary=domainSize)
    galerkinMethodObject.setRHSFunctional([functional, boundaryFunctional2,
                        boundaryFunctional0, boundaryFunctional1])
    # galerkinMethodObject.setRHSFunctional([functional])
    boundaryConditions = ['{"boundaryPoint": "0.0", "boundaryValue": 1.0}',
                          '{"boundaryPoint": "10.0", "boundaryValue": 1.0}']
    # boundaryConditions = []
    galerkinMethodObject.setDirichletBoundaryConditions(boundaryConditions)

    mesh = MeshClass.mesh(1)
    mesh.generateUniformMeshOnRectange([0, domainSize],
                                       [amountOfElements],
                                       [approximationOrder])

    mesh.establishNeighbours()

    mesh.fileWrite("elementsData.txt", "neighboursData.txt")
    mesh.fileRead("elementsData.txt", "neighboursData.txt")
    galerkinMethodObject.initializeMesh(mesh)
    galerkinMethodObject.initializeElements()
    galerkinMethodObject.calculateElements()
    galerkinMethodObject.solveSLAE()

    galerkinMethodObject.checkPositiveEigenvalues()


    w, grid = integr.reg_22_wn(0.0, domainSize, integrationPointsAmount)

    gridSolution = galerkinMethodObject.evaluateSolution(grid)
    # print(gridSolution[0], gridSolution[-1])
    print(galerkinMethodObject.solution[0] - 10.0,
          galerkinMethodObject.solution[-1] - 1.0)
    # plt.plot(grid, gridSolution, label="approximation")
    # plt.plot(grid, np.cos(np.pi * grid), label="exact solution")
    # plt.legend()
    # plt.show()


fun(50, 1, integrationPointsAmount=10000)