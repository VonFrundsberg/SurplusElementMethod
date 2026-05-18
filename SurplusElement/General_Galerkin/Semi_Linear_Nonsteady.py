import matplotlib
import numpy as np

import SurplusElement.Basic_Galerkin.Mesh.mesh as Mesh
from SurplusElement.Basic_Galerkin import Galerkin1d as Galerkin
import SurplusElement.Basic_Galerkin.element.Element1d.element1dUtils as elem1dUtils
import scipy.sparse.linalg as sparse_linalg
import time as time
from collections.abc import Callable
import matplotlib.pyplot as plt

"""
We solve the equation of the form
d(A(u, x)du/dx)/dx + d(B(u, x) u)/dx + C(u, x) u = f,
with zero dirichlet BC
where A, B, C are nonlinear coefficients
"""
class Semi_Linear_Steady_Galerkin:
    def __init__(self, t_start: float, t_finish: float, spatial_integration_points_amount: int) -> None:
        self.__galerkin_method_spatial = Galerkin.GalerkinMethod1d("LS")
        self.__galerkin_method_time = Galerkin.GalerkinMethod1d("LS")
        self.__t_start = t_start
        self.__t_finish = t_finish
        self.__spatial_integration_points_amount = spatial_integration_points_amount

    def set_bilinear_form(self, inner_forms: list, discontinuity_forms: list = [],
                        boundary_forms: list = [], functionals: list = []) -> None:
        self.__inner_forms = inner_forms
        self.__discontinuity_forms = discontinuity_forms
        self.__boundary_forms = boundary_forms
        self.__functionals = functionals

    def set_mesh(self, mesh: Mesh) -> None:
        self.__mesh = mesh

    def grad_form(self, nonlinear_weight: Callable, solution: Callable):
        """nonlinear_weight is expected to be a function of two arguments
           nonlinear_weight(u, x), where u is the solution and x is the independent variable
           solution is substituted as u into nonlinear_weight(u, x)"""
        return lambda trialElement, testElement: elem1dUtils.integrateBilinearForm1(
            trialElement, testElement, lambda x: nonlinear_weight(solution(x), x),
            self.__spatial_integration_points_amount)

    def flux_form(self, nonlinear_weight: Callable, solution: Callable):
        """nonlinear_weight is expected to be a function of two arguments
                   nonlinear_weight(u, x), where u is the solution and x is the independent variable
                   solution is substituted as u into nonlinear_weight(u, x)"""
        return lambda trialElement, testElement: elem1dUtils.integrateBilinearForm2(
            trialElement, testElement, lambda x: nonlinear_weight(solution(x), x),
            self.__spatial_integration_points_amount)

    def reaction_form(self, nonlinear_weight: Callable, solution: Callable):
        """nonlinear_weight is expected to be a function of two arguments
                           nonlinear_weight(u, x), where u is the solution and x is the independent variable
                           solution is substituted as u into nonlinear_weight(u, x)"""
        return lambda trialElement, testElement: elem1dUtils.integrateBilinearForm0(
            trialElement, testElement, lambda x: nonlinear_weight(solution(x), x),
            self.__spatial_integration_points_amount)

    def central_flux_form(self, nonlinear_weight: Callable, solution: Callable, physical_boundary):

        return lambda trialElement, testElement: elem1dUtils.evaluateDG_centralFlux(
            trialElement=trialElement, testElement=testElement, weight= lambda x: nonlinear_weight(solution(x), x),
            physicalBoundary=physical_boundary)

    def boundary_flux_form(self, nonlinear_weight: Callable, solution: Callable, boundary):
        return lambda trialElement, testElement: elem1dUtils.evaluateBilinearFormAtBoundary1(
            trialElement=trialElement, testElement=testElement, weight=lambda x: nonlinear_weight(solution(x), x),
            B=boundary)

    def diffusion_main_discontinuity_form(self, nonlinear_weight: Callable, solution: Callable, physical_boundary):
        return lambda trialElement, testElement: elem1dUtils.evaluateDG_JumpComponentMain(
            trialElement=trialElement, testElement=testElement, weight= lambda x: nonlinear_weight(solution(x), x),
            physicalBoundary=physical_boundary)

    def diffusion_symmetry_discontinuity_form(self, nonlinear_weight: Callable, solution: Callable, physical_boundary):
        return lambda trialElement, testElement: elem1dUtils.evaluateDG_JumpComponentSymmetry(
            trialElement=trialElement, testElement=testElement, weight=lambda x: nonlinear_weight(solution(x), x),
            physicalBoundary=physical_boundary)

    def penalty_discontinuity_form(self, nonlinear_weight: Callable, solution: Callable, physical_boundary):
        return lambda trialElement, testElement: elem1dUtils.evaluateDG_ErrorComponent(
            trialElement=trialElement, testElement=testElement, weight=lambda x: nonlinear_weight(solution(x), x),
            physicalBoundary=physical_boundary)

    def set_dirichlet_boundary_conditions(self, boundary_conditions: list = []) -> None:
        self.__boundary_conditions = boundary_conditions

    def run_implicit_euler(self, time_step: float, store_values: bool = True, eps : float = 1e-10) -> None:
        tau = time_step
        K = self.__time_matrix - tau * self.__spatial_matrix
        ind = (K.getnnz(1) > 0).copy()
        K = K[K.getnnz(1) > 0, :][:, K.getnnz(0) > 0]
        previous_vector =  self.__vectorize_initial_condition()
        LU = sparse_linalg.splu(K)
        time_elapsed = self.__t_start
        solutions = []
        times = []
        if store_values:
            solutions.append(previous_vector)
            times.append(0.0)
        while time_elapsed <= self.__t_finish - eps:
            solution = LU.solve((self.__time_matrix @ previous_vector)[ind])
            new_vector = np.zeros(ind.shape, dtype=float)
            new_vector[ind] = solution
            time_elapsed += tau
            if store_values:
                solutions.append(new_vector)
                times.append(time_elapsed)
            previous_vector = new_vector.copy()

        if store_values:
            self.__solution = np.array(solutions, dtype=float)
            self.__time_slices = np.array(times, dtype=float)
        else:
            self.__solution = np.array(new_vector, dtype=float)

    def set_initial_approximation(self, initial_approximation: Callable | np.ndarray) -> None:
        if callable(initial_approximation):
            self.__galerkin_method.initializeMesh(mesh=self.__mesh)
            self.__galerkin_method.initializeElements()
            grid = self.__galerkin_method.getMeshPoints()
            self.__galerkin_method.solution = initial_approximation(grid)
        else:
            self.__galerkin_method.solution = initial_approximation

    def initialize_elements(self) -> None:
        """initialize bilinear forms"""
        self.__galerkin_method_spatial.setBilinearForm(
            self.__inner_forms, self.__discontinuity_forms, self.__boundary_forms)

        reaction_form = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm0(
            trialElement, testElement, lambda x: 0.0 * x + 1.0, self.__spatial_integration_points_amount)

        self.__galerkin_method_time.setBilinearForm(
            [reaction_form], [], [])

        """initialize functionals"""
        self.__galerkin_method_spatial.setRHSFunctional(self.__functionals)
        self.__galerkin_method_time.setRHSFunctional(self.__functionals)

        """initialize dirichlet boundary conditions"""
        self.__galerkin_method_spatial.setDirichletBoundaryConditions(self.__boundary_conditions)
        self.__galerkin_method_time.setDirichletBoundaryConditions(self.__boundary_conditions)

        """initialize mesh"""
        self.__galerkin_method_spatial.initializeMesh(mesh=self.__mesh)
        self.__galerkin_method_time.initializeMesh(mesh=self.__mesh)

        """initialize elements"""
        self.__galerkin_method_spatial.initializeElements()
        self.__galerkin_method_time.initializeElements()

        """calculate elements"""
        self.__galerkin_method_spatial.calculateElements()
        self.__galerkin_method_time.calculateElements()

        """get matrices for the time evolution process"""
        self.__time_matrix = self.__galerkin_method_time.getMatrices()
        self.__spatial_matrix = self.__galerkin_method_spatial.getMatrices()


    def get_grid(self):
        return self.__galerkin_method.getMeshPoints()

    def solve(self, absolute_error: float, max_iteration: int):

        # np.set_printoptions(precision=2, suppress=True)
        verbose: bool = False
        for i in range(max_iteration):
            M = self.__galerkin_method.getMatrices()
            ind = (M.getnnz(1) > 0).copy()
            M = M[M.getnnz(1) > 0, :][:, M.getnnz(0) > 0]
            solution = sparse_linalg.spsolve(M, self.rhs[ind])

            if verbose:
                print(i, np.max(np.abs(self.__galerkin_method.solution[ind] - solution)))
            if np.max(np.abs(self.__galerkin_method.solution[ind] - solution)) < absolute_error:
                self.__galerkin_method.solution[ind] = solution
                self.__galerkin_method.solution[~ind] *= 0.0
                if verbose:
                    print("solved in ", i, " iterations")
                return
            else:
                self.__galerkin_method.solution[ind] = solution
                self.__galerkin_method.solution[~ind] *= 0.0
                self.__galerkin_method.calculateElements()
        if verbose:
            print("limit of ", max_iteration, " has been reached")

    def evaluate_solution_on_grid_over_time_interval(self, t_start: float, t_finish: float, points: np.ndarray) -> np.ndarray:
        eligible_indices = np.where((self.__time_slices >= t_start) & (self.__time_slices <= t_finish))
        to_evaluate = self.__solution[eligible_indices]
        result = self.__galerkin_method_spatial.evaluateFunctionsAtPoints(to_evaluate.T, points).T
        return result

    def __vectorize_initial_condition(self) -> np.ndarray:
        grid = self.__galerkin_method_spatial.getMeshPoints()
        result = self.__initial_condition(grid)
        return result