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
    def __init__(self, integration_points_amount: int) -> None:
        self.__galerkin_method = Galerkin.GalerkinMethod1d("LS")
        self.__spatial_integration_points_amount = integration_points_amount

    def set_bilinear_form(self, inner_forms: list, discontinuity_forms: list = [],
                        boundary_forms: list = [], functionals: list = []) -> None:
        self.__inner_forms = inner_forms
        self.__discontinuity_forms = discontinuity_forms
        self.__boundary_forms = boundary_forms
        self.__functionals = functionals

    def set_mesh(self, mesh: Mesh) -> None:
        self.__mesh = mesh
    def evaluate_solution(self, x: np.ndarray) -> np.ndarray:
        return self.__galerkin_method.evaluateSolution(x)
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
        self.__galerkin_method.setBilinearForm(
            self.__inner_forms, self.__discontinuity_forms, self.__boundary_forms)

        """initialize functionals"""
        self.__galerkin_method.setRHSFunctional(self.__functionals)

        """initialize dirichlet boundary conditions"""
        self.__galerkin_method.setDirichletBoundaryConditions(self.__boundary_conditions)

        """initialize mesh"""
        self.__galerkin_method.initializeMesh(mesh=self.__mesh)

        """initialize elements"""
        self.__galerkin_method.initializeElements()

        """calculate elements"""
        self.__galerkin_method.calculateElements()

        """get matrices for the time evolution process"""
        self.matrix = self.__galerkin_method.getMatrices()
        self.rhs = self.__galerkin_method.getRHS()


    def get_grid(self):
        return self.__galerkin_method.getMeshPoints()

    def solve(self, absolute_error: float, max_iteration: int):
        np.set_printoptions(precision=2, suppress=True)

        for i in range(max_iteration):
            M = self.__galerkin_method.getMatrices()
            ind = (M.getnnz(1) > 0).copy()
            M = M[M.getnnz(1) > 0, :][:, M.getnnz(0) > 0]
            solution = sparse_linalg.spsolve(M, self.rhs[ind])
            print(i, np.max(np.abs(self.__galerkin_method.solution[ind] - solution)))
            if np.max(np.abs(self.__galerkin_method.solution[ind] - solution)) < absolute_error:
                self.__galerkin_method.solution[ind] = solution
                self.__galerkin_method.solution[~ind] *= 0.0
                print("solved in ", i, " iterations")
                return
            else:
                self.__galerkin_method.solution[ind] = solution
                self.__galerkin_method.solution[~ind] *= 0.0
                self.__galerkin_method.calculateElements()
        print("limit of ", max_iteration, " has been reached")

