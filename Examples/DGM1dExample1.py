import numpy as np
from matplotlib import legend

from SurplusElement.GalerkinMethod import Galerkin1d as galerkin
import SurplusElement.GalerkinMethod.element.Element1d.element1dUtils as elem1dUtils
import SurplusElement.GalerkinMethod.Mesh.mesh as MeshClass

from SurplusElement.mathematics import integrate as integr
import matplotlib.pyplot as plt

"""

The problem is 
d2u/dx^2 = sin(pi * x) on the [-1, 1] interval with zero dirichlet BC which are enforced strongly
Mesh points and approximation orders are uniform and 
being controlled by amountOfElements and approximationOrder respectively
"""
def fun(approximationOrder, amountOfElements, integrationPointsAmount = 500):
    galerkinMethodObject = galerkin.GalerkinMethod1d("LS")

    gradForm = "integral du/dx * dv/dx dx"
    boundaryForm1 = "boundaryIntegral [u] <dv/dx>"
    boundaryForm2 = "boundaryIntegral [v] <du/dx)>"

    gradForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm1(
        trialElement, testElement, lambda x: x * 0.0 - 1.0, integrationPointsAmount)

    def boundaryForm1(trialElement: galerkin.element.Element1d, elementTest: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_JumpComponentMain(
            trialElement=trialElement, testElement=elementTest, weight=lambda x: x * 0.0 + 1.0)

    def boundaryForm2(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_JumpComponentSymmetry(
            trialElement=trialElement, testElement=testElement, weight=lambda x: x * 0.0 + 1.0)

    functional = "integral sin(pi x) u(x) f(x) dx"

    functional = lambda testElement: elem1dUtils.integrateFunctional(
        testElement=testElement, function=lambda x: np.sin(np.pi * x),
            weight=lambda x: x * 0.0 + 1.0, integrationPointsAmount=integrationPointsAmount)

    galerkinMethodObject.setBilinearForm(innerForms=[gradForm], discontinuityForms=[boundaryForm1, boundaryForm2])
    galerkinMethodObject.setRHSFunctional([functional])
    boundaryConditions = ['{"boundaryPoint": "-1.0", "boundaryValue": 0.0}',
                          '{"boundaryPoint": "1.0", "boundaryValue": 0.0}']

    galerkinMethodObject.setDirichletBoundaryConditions(boundaryConditions)

    mesh = MeshClass.mesh(1)
    mesh.generateUniformMeshOnRectange([-1.0, 1.0],
                                       [amountOfElements],
                                       [approximationOrder])

    mesh.establishNeighbours()

    mesh.fileWrite("elementsData.txt", "neighboursData.txt")
    mesh.fileRead("elementsData.txt", "neighboursData.txt")
    galerkinMethodObject.initializeMesh(mesh)
    galerkinMethodObject.initializeElements()
    galerkinMethodObject.calculateElements()
    galerkinMethodObject.solveSLAE()


    w, grid = integr.reg_22_wn(-1.0, 1.0, integrationPointsAmount)

    gridSolution = galerkinMethodObject.evaluateSolutionAtPoints(grid)

    plt.plot(grid, gridSolution, label="approximation")
    plt.plot(grid, -np.pi**(-2) * np.sin(np.pi * grid), label="exact solution")
    # plt.plot(grid, np.sin(np.pi * grid), label="right hand side")
    plt.legend()
    plt.show()


fun(3, 20, integrationPointsAmount=10000)