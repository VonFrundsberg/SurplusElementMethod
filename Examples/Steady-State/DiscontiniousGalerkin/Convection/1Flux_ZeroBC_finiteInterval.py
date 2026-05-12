import numpy as np
from matplotlib import legend

from SurplusElement.Basic_Galerkin import Galerkin1d as galerkin
import SurplusElement.Basic_Galerkin.element.Element1d.element1dUtils as elem1dUtils
import SurplusElement.Basic_Galerkin.Mesh.mesh as MeshClass

from SurplusElement.mathematics import integrate as integr
import matplotlib.pyplot as plt


"""
The problem is 
d(u/dx) = sin(x) on the [0, pi] interval with dirichlet BC set to u(0) = 0.0
"""
def fun(approximationOrder, amountOfElements, integrationPointsAmount = 500):
    galerkinMethodObject = galerkin.GalerkinMethod1d("LS")

    fluxForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm2(
        trialElement, testElement, lambda x: 0 * x - 1.0, integrationPointsAmount)

    def fluxDGForm(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_centralFlux(
            trialElement=trialElement, testElement=testElement, weight=lambda x: 0 * x + 1.0,
            physicalBoundary=np.array([0.0, np.pi]))

    def fluxBoundaryForm(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateBilinearFormAtBoundary1(
            trialElement=trialElement, testElement=testElement, weight=lambda x: 0.0 * x + 1.0,
            B=np.pi)


    functional = lambda testElement: elem1dUtils.integrateFunctional(
        testElement=testElement, function=lambda x: np.sin(x),
            weight=lambda x: x * 0.0 + 1.0, integrationPointsAmount=integrationPointsAmount)


    galerkinMethodObject.setBilinearForm(innerForms=[fluxForm],
                                         discontinuityForms=[fluxDGForm],
                                         boundaryForms=[fluxBoundaryForm]
                                         )

    galerkinMethodObject.setRHSFunctional([functional])
    boundaryConditions = ['{"boundaryPoint": "0.0", "boundaryValue": 0.0}',
                          '{"boundaryPoint": "np.pi", "boundaryValue": 1.0}']
    # boundaryConditions = []
    galerkinMethodObject.setDirichletBoundaryConditions(boundaryConditions)

    mesh = MeshClass.mesh(1)
    mesh.generateUniformMeshOnRectange([0.0, np.pi],
                                       [amountOfElements],
                                       [approximationOrder])

    mesh.establishNeighbours()

    mesh.fileWrite("elementsData.txt", "neighboursData.txt")
    mesh.fileRead("elementsData.txt", "neighboursData.txt")
    galerkinMethodObject.initializeMesh(mesh)
    galerkinMethodObject.initializeElements()
    galerkinMethodObject.calculateElements()
    galerkinMethodObject.solveSLAE()

    # galerkinMethodObject.checkPositiveEigenvalues()


    w, grid = integr.reg_22_wn(0.0, np.pi, integrationPointsAmount)

    gridSolution = galerkinMethodObject.evaluateSolution(grid)
    analyticSolution = lambda x: (1.0 - np.cos(x))
    # print(gridSolution[0], gridSolution[-1])
    # print(galerkinMethodObject.solution[0] - 10.0,
    #       galerkinMethodObject.solution[-1] - 1.0)

    # plt.plot(grid, gridSolution - analyticSolution(grid), label="approximation")
    print(approximationOrder, np.max(np.abs(gridSolution - analyticSolution(grid))))
    # print(approximationOrder, np.max(np.abs(gridSolution - analyticSolution(grid))))
    # plt.show()
    # plt.plot(grid, gridSolution, label="approximation")
    # plt.plot(grid, analyticSolution(grid), label="exact solution")
    # plt.legend()
    # plt.show()

import time as time
for i in range(2, 21):
    fun(i, 10, integrationPointsAmount=10000)
    # time.sleep(5)