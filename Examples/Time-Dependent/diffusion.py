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
du/dt - d2u/dx2 = 0 on the [0, pi] interval with IC
u(x, 0) = sin(x)
and BC
u(pi, t) = 0
u(0, t) = 0
"""
def fun(approximationOrder, integrationPointsAmount = 500):

    gradForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm1(
        trialElement, testElement, lambda x: np.nan_to_num(x=x * 0.0, nan=0.0) - 1.0, integrationPointsAmount)

    functional = lambda testElement: elem1dUtils.integrateFunctional(
        testElement=testElement, function=lambda x: x * 0.0 + 0.0,
        weight=lambda x: x * 0.0 + 1.0, integrationPointsAmount=integrationPointsAmount)
    boundaryConditions = ['{"boundaryPoint": "0.0", "boundaryValue": 0.0}',
                          '{"boundaryPoint": "np.pi", "boundaryValue": 0.0}']
    mesh = MeshClass.mesh(1)
    mesh.generateUniformMeshOnRectange([0.0, np.pi],
                                       [1],
                                       [approximationOrder])

    mesh.establishNeighbours()
    mesh.fileWrite("elementsData.txt", "neighboursData.txt")
    mesh.fileRead("elementsData.txt", "neighboursData.txt")


    initial_condition = lambda x: np.sin(x)


    t_start = 0.0
    t_finish = 1.0
    time_step = 0.01
    numericalPDE = bilinTDP(t_start=t_start, t_finish=t_finish,
                            spatial_integration_points_amount=integrationPointsAmount)
    numericalPDE.set_bilinear_form(inner_forms=[gradForm], boundary_forms=[],
                                   discontinuity_forms=[],functionals=[functional])
    numericalPDE.set_dirichlet_boundary_conditions(boundary_conditions=boundaryConditions)
    numericalPDE.set_mesh(mesh=mesh)
    numericalPDE.set_initial_condition(initial_condition=initial_condition)
    numericalPDE.initialize_elements()
    numericalPDE.run_implicit_euler(time_step=time_step)

    w, grid = integr.reg_22_wn(0.0, np.pi, int((t_finish - t_start)/time_step))
    evaluated_solution = numericalPDE.evaluate_solution_on_grid_over_time_interval(t_start=t_start, t_finish=t_finish, points=grid)
    """
    In the case of sin(x) as IC on [0, pi] interval,
     the analytic solution is IC multiplied by the e^(-t)
    """
    print("maximum absolute error at t = " + str(t_finish) + ": ",
          np.max(np.abs(evaluated_solution[-1, :] - np.sin(grid) * np.exp(-t_finish))))
    def analyticSolution(x):
        return initial_condition(x) * np.exp(-t_finish)

    plt.plot(grid, evaluated_solution[-1, :], label="numerical solution at t_finish")
    plt.plot(grid, analyticSolution(grid), label="analytic solution at t_finish")
    plt.legend()
    plt.show()


for i in range(6, 21):
    fun(i,  integrationPointsAmount=100000)