import numpy as np
from matplotlib import legend

from SurplusElement.GalerkinMethod import Galerkin1d as galerkin
import SurplusElement.GalerkinMethod.element.Element1d.element1dUtils as elem1dUtils
import SurplusElement.GalerkinMethod.Mesh.mesh as MeshClass

from SurplusElement.mathematics import integrate as integr
import matplotlib.pyplot as plt


"""

The problem is 
-d2u/dx^2 + u(x) = exp(-x) + 1 on the [0, 10] interval with dirichlet BC set to u(0) = 10, u(10) = 1
The solution is 
-\frac{e^{-x} \left(-e^{20} x+x+2 e^x+28 e^{2 x}-2 e^{x+20}-18 e^{20}-10\right)}{2 \left(e^{20}-1\right)}
u(x) = -1 * 
(np.exp(-x) * (-10.0 -18.0 * np.exp(20.0) + 2.0 * np.exp(x) +
 28 * np.exp(2.0 * x) - 2 * np.exp(20.0 + x) + x - np.exp(20.0) * x)
/(2.0 * (-1.0 + np.exp(20.0))
with weak dirichlet boundary conditions enforcement
Mesh points and approximation orders are uniform and 
being controlled by amountOfElements and approximationOrder respectively
"""
def fun(approximationOrder, amountOfElements, integrationPointsAmount = 500):
    domainSize = 10.0
    galerkinMethodObject = galerkin.GalerkinMethod1d("LS")

    gradForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm1(
        trialElement, testElement, lambda x: x * 0.0 + 1.0, integrationPointsAmount)

    innerForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm0(
        trialElement, testElement, lambda x: x * 0.0 + 1.0, integrationPointsAmount)
    eps = 1e-15
    def DGForm1(trialElement: galerkin.element.Element1d, elementTest: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_JumpComponentMain(
            trialElement=trialElement, testElement=elementTest, weight=lambda x: x * 0.0 - 1.0,
            physicalBoundary=np.array([0, domainSize]), eps = eps)

    def DGForm2(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_JumpComponentSymmetry(
            trialElement=trialElement, testElement=testElement, weight=lambda x: x * 0.0 - 1.0,
            physicalBoundary=np.array([0, domainSize]), eps = eps)

    sigma = approximationOrder ** 2 / (10.0 / amountOfElements)

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

    galerkinMethodObject.setBilinearForm(innerForms=[gradForm, innerForm],
                                         discontinuityForms=[DGForm1, DGForm2],
                                         boundaryForms=[boundaryForm1, boundaryForm2,
                                                        boundaryForm3, boundaryForm4]
                                         )
    # galerkinMethodObject.setRHSFunctional([functional, boundaryFunctional0, boundaryFunctional1])
    galerkinMethodObject.setRHSFunctional([functional])
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


    w, grid = integr.reg_22_wn(0.0, domainSize, integrationPointsAmount)

    gridSolution = galerkinMethodObject.evaluateSolutionAtPoints(grid)

    plt.plot(grid, gridSolution, label="approximation")
    # plt.plot(grid, np.cos(np.pi * grid), label="exact solution")
    plt.legend()
    plt.show()


fun(3, 15, integrationPointsAmount=10000)