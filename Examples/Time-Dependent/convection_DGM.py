import numpy as np
from matplotlib import legend
from scipy import special

from SurplusElement.Basic_Galerkin import Galerkin1d as galerkin
import SurplusElement.Basic_Galerkin.element.Element1d.element1dUtils as elem1dUtils
import SurplusElement.Basic_Galerkin.Mesh.mesh as MeshClass

from SurplusElement.mathematics import integrate as integr
import matplotlib.pyplot as plt

from SurplusElement.General_Galerkin.Bilinear_Nonsteady_Galerkin import Bilinear_Nonsteady_Galerkin as bilinTDP
"""
The problem is 
du/dt - du/dx = 0 on the [0, pi] interval with IC
u(x, 0) = sin(x)
and BC
u(0, t) = 0
"""
def fun(approximationOrder, amount_of_elements, integrationPointsAmount = 500):

    fluxForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm2(
        trialElement, testElement, lambda x: np.nan_to_num(x=x * 0.0, nan=0.0) + 1.0, integrationPointsAmount)

    def fluxDGForm(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_centralFlux(
            trialElement=trialElement, testElement=testElement, weight=lambda x: 0 * x - 1.0,
            physicalBoundary=np.array([0.0, np.pi]))

    def fluxBoundaryForm(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateBilinearFormAtBoundary1(
            trialElement=trialElement, testElement=testElement, weight=lambda x: 0.0 * x - 1.0,
            B=np.pi)


    functional = lambda testElement: elem1dUtils.integrateFunctional(
        testElement=testElement, function=lambda x: x * 0.0 + 0.0,
        weight=lambda x: x * 0.0 + 1.0, integrationPointsAmount=integrationPointsAmount)
    boundaryConditions = ['{"boundaryPoint": "0.0", "boundaryValue": 0.0}',
                          '{"boundaryPoint": "np.pi", "boundaryValue": 1.0}']
    mesh = MeshClass.mesh(1)
    mesh.generateUniformMeshOnRectange([0.0, np.pi],
                                       [amount_of_elements],
                                       [approximationOrder])

    mesh.establishNeighbours()
    mesh.fileWrite("elementsData.txt", "neighboursData.txt")
    mesh.fileRead("elementsData.txt", "neighboursData.txt")


    initial_condition = lambda x: np.sin(x)

    t_start = 0.0
    t_finish = 1.0
    time_step = 0.00001

    numericalPDE = bilinTDP(t_start=t_start, t_finish=t_finish,
                            spatial_integration_points_amount=integrationPointsAmount)
    numericalPDE.set_bilinear_form(inner_forms=[fluxForm], boundary_forms=[fluxBoundaryForm],
                                   discontinuity_forms=[fluxDGForm],functionals=[functional])
    numericalPDE.set_dirichlet_boundary_conditions(boundary_conditions=boundaryConditions)
    numericalPDE.set_mesh(mesh=mesh)
    numericalPDE.set_initial_condition(initial_condition=initial_condition)
    numericalPDE.initialize_elements()
    numericalPDE.run_implicit_euler(time_step=time_step)

    w, grid = integr.reg_22_wn(0.0, np.pi, 1000)
    evaluated_solution = numericalPDE.evaluate_solution_on_grid_over_time_interval(0.0, 1.0, points=grid)
    # plt.plot(grid, evaluated_solution[-1, :])
    def analyticSolution(x):
        x = np.atleast_1d(x)
        result = np.zeros(x.shape)
        less_than_zero = np.where(x < 0.0)
        more_than_zero = np.where(x >= 0.0)
        result[less_than_zero] = 0.0
        result[more_than_zero] = np.sin(x[more_than_zero])
        return result

    """
    In the case of sin(x) as IC on [0, pi] interval,
     the analytic solution is IC shifted by t
    """
    print("maximum absolute error at t = " + str(t_finish) + ": ",
          np.max(np.abs(evaluated_solution[-1, :] - analyticSolution(grid - t_finish))))
    # plt.plot(grid, evaluated_solution[0, : ], label="initial condition")
    # plt.plot(grid, evaluated_solution[int((t_finish - t_start) / time_step /2.0), :], label="numerical solution at half time")
    # plt.plot(grid, evaluated_solution[-1, :], label="numerical solution at t_finish")
    # plt.plot(grid, analyticSolution(grid - t_finish), label="analytic solution at t_finish")
    # plt.legend()
    # plt.show()

    plt.plot(grid, evaluated_solution[-1, :] - analyticSolution(grid - t_finish), label="difference of analytic and numerical solution at t_finish")
    plt.legend()
    plt.show()










for i in range(10, 11):
    fun(i, amount_of_elements=10, integrationPointsAmount=10000)