import numpy as np
from matplotlib import legend

from SurplusElement.General_Galerkin.Semi_Linear_Steady_Galerkin import Semi_Linear_Steady_Galerkin as galerkinMethod
import SurplusElement.Basic_Galerkin.element.Element1d.element1dUtils as elem1dUtils
import SurplusElement.Basic_Galerkin.Mesh.mesh as MeshClass

from SurplusElement.mathematics import integrate as integr
import matplotlib.pyplot as plt

"""
The problem is 
d/dx((u**2 + 1) du/dx) = cos(2 x) on the [0, pi] interval with zero dirichlet BC which are enforced strongly
"""
def fun(approximationOrder, amountOfElements, integrationPointsAmount = 500):
    galerkinMethodObject = galerkinMethod(integrationPointsAmount)
    domain_size = np.pi
    gradForm = galerkinMethodObject.grad_form(nonlinear_weight =
                                              lambda u, x: (u**2 + 1.0), solution=galerkinMethodObject.evaluate_solution)

    sigma = 1 * approximationOrder ** 2 / (domain_size / amountOfElements)

    dg_form_1 = galerkinMethodObject.diffusion_main_discontinuity_form(nonlinear_weight =
        lambda u, x: -(u**2 + 1.0), solution=galerkinMethodObject.evaluate_solution,
                                                                       physical_boundary=np.array([0, domain_size]))
    dg_form_2 = galerkinMethodObject.diffusion_symmetry_discontinuity_form(nonlinear_weight =
        lambda u, x: -(u**2 + 1.0), solution=galerkinMethodObject.evaluate_solution,
                                                                       physical_boundary=np.array([0, domain_size]))
    dg_form_3 = galerkinMethodObject.penalty_discontinuity_form(nonlinear_weight =
        lambda u, x: sigma * (u**2 + 1.0), solution=galerkinMethodObject.evaluate_solution,
                                                                       physical_boundary=np.array([0, domain_size]))
    functional = lambda testElement: elem1dUtils.integrateFunctional(
        testElement=testElement, function=lambda x: -0.5 * (-1.0 + 3.0 * np.cos(2.0 * x)) * np.sin(x),
        weight=lambda x: x * 0.0 + 1.0, integrationPointsAmount=integrationPointsAmount)

    galerkinMethodObject.set_bilinear_form(inner_forms=[gradForm],
                                           discontinuity_forms=[dg_form_1, dg_form_2, dg_form_3],
                                           functionals=[functional])
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

for i in range(3, 11):
    fun(i, 4, integrationPointsAmount=10000)