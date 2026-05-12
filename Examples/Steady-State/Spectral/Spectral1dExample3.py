import numpy as np
from matplotlib import legend
from scipy import special

from SurplusElement.Basic_Galerkin import Galerkin1d as galerkin
import SurplusElement.Basic_Galerkin.element.Element1d.element1dUtils as elem1dUtils
import SurplusElement.Basic_Galerkin.Mesh.mesh as MeshClass

from SurplusElement.mathematics import integrate as integr
import matplotlib.pyplot as plt


"""
The problem is 
-d2u/dx^2 - d(x * u(x)) + u(x) = exp(-x^2) on the [0, inf] interval with dirichlet BC set to u(0) = 0, u(inf) = 0
BC are included into the space
"""
def fun(approximationOrder, integrationPointsAmount = 500):
    galerkinMethodObject = galerkin.GalerkinMethod1d("LS")

    gradForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm1(
        trialElement, testElement, lambda x: np.nan_to_num(x=x * 0.0, nan=0.0) + 1.0, integrationPointsAmount)
    fluxForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm2(
        trialElement, testElement, lambda x: x, integrationPointsAmount)
    innerForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm0(
        trialElement, testElement, lambda x: np.nan_to_num(x=x * 0.0, nan=0.0) + 1.0, integrationPointsAmount)

    functional = lambda testElement: elem1dUtils.integrateFunctional(
        testElement=testElement, function=lambda x: (np.exp(-x**2)),
            weight=lambda x: np.nan_to_num(x=x * 0.0, nan=0.0) + 1.0, integrationPointsAmount=integrationPointsAmount)

    galerkinMethodObject.setBilinearForm(innerForms=[gradForm, innerForm, fluxForm],
                                         discontinuityForms=[],
                                         boundaryForms=[]
                                         )
    galerkinMethodObject.setRHSFunctional([functional])
    boundaryConditions = ['{"boundaryPoint": "0.0", "boundaryValue": 0.0}',
                          '{"boundaryPoint": "np.inf", "boundaryValue": 0.0}']
    galerkinMethodObject.setDirichletBoundaryConditions(boundaryConditions)

    mesh = MeshClass.mesh(1)
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


    w, grid = integr.reg_22_wn(0.0, 10.0, integrationPointsAmount)

    gridSolution = galerkinMethodObject.evaluateSolution(grid)
    analyticSolution = lambda x: np.pi * 0.25 * special.erf(grid/np.sqrt(2.0)) * (1.0 - special.erf(grid/np.sqrt(2.0)))
    gridSol = galerkinMethodObject.evaluateSolution(grid)
    d1gridSol = galerkinMethodObject.evaluateSolutionDerivative(grid)
    d2gridSol = galerkinMethodObject.evaluateSolutionDerivative(grid, 2)

    print(approximationOrder, np.max(np.abs(gridSolution - analyticSolution(grid))))
    # plt.plot(grid, -d2gridSol - grid * d1gridSol, label="lhs")
    # plt.plot(grid, np.exp(-grid**2))
    # plt.show()
    # print(i, galerkinMethodObject.solution[0] - 9.0, galerkinMethodObject.solution[-1] - 0.0)
    # print(i, np.abs(galerkinMethodObject.solution[0] - 9.0))
    # print(i, np.max(np.abs(-d2gridSol + gridSol - np.exp(-grid))))
    # plt.plot(grid,d2gridSol)
    # plt.plot(grid, analyticSolutionD2(grid))
    # print(i,
    #       np.max(np.abs(gridSol - analyticSolution(grid)))/np.max(np.abs(analyticSolution(grid))),
    #       np.max(np.abs(d2gridSol - analyticSolutionD2(grid)))/np.max(np.abs(analyticSolutionD2(grid))))
    # print(i, np.sqrt(w @ (gridSol - analyticSolution(grid))**2) / np.sqrt(w @ (analyticSolution(grid))**2),
    #       np.sqrt(w @ (d2gridSol - analyticSolutionD2(grid))**2) / np.sqrt(w @ (analyticSolutionD2(grid))**2),
    #       np.sqrt(w @ (-d2gridSol + gridSol - np.exp(-grid))**2) / np.sqrt(w @ (np.exp(-grid))**2))
    # plt.plot(grid, gridSolution, label="approximation")
    # plt.plot(grid, analyticSolution(grid), label="exact solution")
    # plt.legend()
    # plt.show()
    # plt.plot(grid, -d2gridSol + gridSol, label="numerical LHS")
    # plt.plot(grid, np.exp(-grid), label="exact RHS")
    # plt.show()

for i in range(3, 51):
    fun(i, integrationPointsAmount=100000)