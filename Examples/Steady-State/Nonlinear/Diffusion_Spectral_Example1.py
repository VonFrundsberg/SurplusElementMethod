import numpy as np
from matplotlib import legend

from SurplusElement.General_Galerkin.Semi_Linear_Steady_Galerkin import Semi_Linear_Steady_Galerkin as galerkinMethod
import SurplusElement.Basic_Galerkin.element.Element1d.element1dUtils as elem1dUtils
import SurplusElement.Basic_Galerkin.Mesh.mesh as MeshClass

from SurplusElement.mathematics import integrate as integr
import matplotlib.pyplot as plt

"""
The problem is 
d/dx(u du/dx) = cos(2 x) on the [0, pi] interval with zero dirichlet BC which are enforced strongly
"""
def fun(approximationOrder, amountOfElements, integrationPointsAmount = 500):
    galerkinMethodObject = galerkinMethod(integrationPointsAmount)

    gradForm = galerkinMethodObject.grad_form(nonlinear_weight = lambda u, x: (u**2 + 1.0), solution=galerkinMethodObject.evaluate_solution)

    # def boundaryForm1(trialElement: galerkin.element.Element1d, elementTest: galerkin.element.Element1d):
    #     return elem1dUtils.evaluateDG_JumpComponentMain(
    #         trialElement=trialElement, testElement=elementTest, weight=lambda x: x * 0.0 + 1.0, physicalBoundary=[-1.0, 1.0])
    #
    # def boundaryForm2(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
    #     return elem1dUtils.evaluateDG_JumpComponentSymmetry(
    #         trialElement=trialElement, testElement=testElement, weight=lambda x: x * 0.0 + 1.0, physicalBoundary=[-1.0, 1.0])

    # functional = lambda testElement: elem1dUtils.integrateFunctional(
    #     testElement=testElement, function=lambda x: np.cos(2 * x),
    #         weight=lambda x: x * 0.0 + 1.0, integrationPointsAmount=integrationPointsAmount)
    functional = lambda testElement: elem1dUtils.integrateFunctional(
        testElement=testElement, function=lambda x: -0.5 * (-1.0 + 3.0 * np.cos(2.0 * x)) * np.sin(x),
        weight=lambda x: x * 0.0 + 1.0, integrationPointsAmount=integrationPointsAmount)

    galerkinMethodObject.set_bilinear_form(inner_forms=[gradForm], functionals=[functional])
    boundaryConditions = ['{"boundaryPoint": "0.0", "boundaryValue": 0.0}',
                          '{"boundaryPoint": "np.pi", "boundaryValue": 0.0}']

    galerkinMethodObject.set_dirichlet_boundary_conditions(boundaryConditions)

    mesh = MeshClass.mesh(1)
    mesh.generateUniformMeshOnRectange([0, np.pi],
                                       [amountOfElements],
                                       [approximationOrder])

    mesh.establishNeighbours()

    mesh.fileWrite("elementsData.txt", "neighboursData.txt")
    mesh.fileRead("elementsData.txt", "neighboursData.txt")
    galerkinMethodObject.set_mesh(mesh)
    galerkinMethodObject.set_initial_approximation(initial_approximation = lambda x: 0.0 * x + 1.0)
    galerkinMethodObject.initialize_elements()
    galerkinMethodObject.solve(absolute_error=1e-10, max_iteration=100)

    w, grid = integr.reg_22_wn(0, np.pi, integrationPointsAmount)

    gridSolution = galerkinMethodObject.evaluate_solution(grid)
    analyticSolution = np.sin(grid)
    # plt.plot(grid, gridSolution, label="approximation")
    # plt.plot(grid, analyticSolution, label="exact solution")
    # plt.plot(grid, np.sin(np.pi * grid), label="right hand side")
    # plt.legend()
    # plt.show()
    print(approximationOrder, np.max(np.abs(gridSolution - analyticSolution)))

for i in range(3, 21):
    fun(i, 1, integrationPointsAmount=10000)