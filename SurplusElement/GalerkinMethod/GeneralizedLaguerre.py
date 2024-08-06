import numpy as np
import scipy.linalg as sp_linalg
import scipy.special as sp_spec
from SurplusElement.GalerkinMethod.element.Element1d.AuxClasses import ApproximationSpaceParameter
import json
from typing import Callable


class GeneralizedLaguerre:
    approximationSpaceParameter: ApproximationSpaceParameter
    def __init__(self, approximationOrder:int, alpha: float, nucleusCharge:int):
        self.__approximationSpaceParameter = ApproximationSpaceParameter()
        self.__approxOrder = approximationOrder
        self.__alpha = alpha
        self.__nodes2, self.__weights2 = sp_spec.roots_genlaguerre(approximationOrder, 2)
        self.__nodes1, self.__weights1 = sp_spec.roots_genlaguerre(approximationOrder, 1)
        self.nucleusCharge = nucleusCharge
        n = self.__approxOrder
        a = self.__alpha
        x = self.__nodes2
        nRange = np.arange(start=0, stop=n, step=1)
        self.__evaluatedPolys2 = np.asarray(list(map(lambda n: sp_spec.eval_genlaguerre(n, 2, x), nRange))).T
        self.__M2 = np.einsum('ij,ik->ijk', self.__evaluatedPolys2, self.__evaluatedPolys2)

    def getWeightsNodes(self):
        return self.__weights2, self.__nodes2
    def setRHSFunctional(self, functionals):
        self.functionals = functionals

    def setApproximationSpaceParameters(self, parameters):
        approximationSpaceParameter = ApproximationSpaceParameter()
        parsedJsonParameterInfo = json.loads(parameters)

        context = {
            'np': np
        }

        if "s" in parsedJsonParameterInfo:
            approximationSpaceParameter.s = eval(parsedJsonParameterInfo["s"], context)

        self.approximationSpaceParameter = approximationSpaceParameter


    def recalculateElements_EIG(self, potentialFuncs: list[Callable]):
       i = 2
       x2 = self.__nodes2
       w2 = self.__weights2
       for potential in potentialFuncs:
           self.__lhsMatrixElements[i] = np.einsum('ijk, i -> jk', self.__M2, w2 * potential(x2))
           i += 1

    def calculateElementsDenseEig(self, potentialFuncs: list[Callable]):
        """
        """

        innerFormsAmount = len(potentialFuncs) + 2
        rhsFormsAmount = 1
        self.__lhsMatrixElements = [None] * innerFormsAmount
        self.__rhsMatrixElements = [None] * rhsFormsAmount
        n = self.__approxOrder
        a = self.__alpha
        x1 = self.__nodes1
        w1 = self.__weights1
        x2 = self.__nodes2
        w2 = self.__weights2
        nRange = np.arange(start=0, stop=n, step=1)

        # np.set_printoptions(precision=3, suppress=True)
        evaluatedPolys = np.asarray(list(map(lambda n: 2 * sp_spec.eval_genlaguerre(n - 1, 3, x2) + sp_spec.eval_genlaguerre(n, 2, x2), nRange))).T
        M = np.einsum('ij,ik->ijk', evaluatedPolys, evaluatedPolys)
        self.__lhsMatrixElements[0] = 0.5 * 0.25 * np.einsum('ijk, i -> jk', M, w2)
        # print(self.__lhsMatrixElements[0])

        evaluatedPolys = np.asarray(list(map(lambda n: sp_spec.eval_genlaguerre(n, 2, x1), nRange))).T
        M = np.einsum('ij,ik->ijk', evaluatedPolys, evaluatedPolys)
        self.__lhsMatrixElements[1] = - self.nucleusCharge * np.einsum('ijk, i -> jk', M, w1)
        self.__rhsMatrixElements[0] = np.einsum('ijk, i -> jk', self.__M2, w2)
        # print(self.__lhsMatrixElements[1])

        i = 2
        for potential in potentialFuncs:
            self.__lhsMatrixElements[i] = np.einsum('ijk, i -> jk', self.__M2, w2 * potential(x2))
            i += 1
        # print(self.__rhsMatrixElements[0])
        # time.sleep(500)
        # for it in self.__lhsMatrixElements:
        #     print(it)


        # print(self.__rhsMatrixElements[0])


    def solveEIG_denseMatrix(self, realize=True, amountOfEigs=1, sumMatrices=True):
        """Only for single-domain case.
           Only for zero Dirichlet or Neumann boundary conditions
            Solves generalized eigenvalue problem:
            lhsMatrixElements @ u = lambda * rhsMatrixElements @ u

            Returns:
                ALL eigenpairs"""
        if sumMatrices:
            # for it in self.__lhsMatrixElements:
                # print(it.shape)
            A = np.sum(np.asarray(self.__lhsMatrixElements), axis=0)
            B = np.sum(np.asarray(self.__rhsMatrixElements), axis=0)
        else:
            A = self.__lhsMatrixElements
            B = self.__rhsMatrixElements
        # np.savetxt(X=A, fname="A.txt", fmt='%1.3f')
        # np.savetxt(X=B, fname="B.txt", fmt='%1.3f')
        # print(B)
        # print(A)
        # print(B)
        # A[-1, :] = 0
        # A[:, -1] = 0
        # B[-1, :] = 0
        # B[:, -1] = 0
        # ind = ~(A==0).all(1)
        # A = A[~(A==0).all(1), :][:, ~(A==0).all(0)]
        # B = B[~(B == 0).all(1), :][:, ~(B == 0).all(0)]
        # print(A)
        # print(B)
        # print(sp_linalg.det(A))
        # print(sp_linalg.det(B))
        # self.A = A
        # self.B = B
        if sp_linalg.issymmetric(A) and sp_linalg.issymmetric(B):
            values, vectors = sp_linalg.eigh(A, B)
        else:
            values, vectors = sp_linalg.eig(A, B)
        if realize:
            eigvalIndices = np.argsort(np.real(values))
            values = np.real(values[eigvalIndices])
            vectors = np.real(vectors[:, eigvalIndices])
        if amountOfEigs == 1:
            self.solutionWithDirichletBC = vectors[:, 0]
        else:
            self.solutionWithDirichletBC = vectors[:, :amountOfEigs]
        return values, vectors


    def evaluateSolutionAtPointsWithExponent(self, x):

        """

        """
        x = np.atleast_1d(x)
        nRange = np.arange(start=0, stop=self.__approxOrder, step=1)
        evaluated_basis = np.asarray(list(map(lambda n: np.exp(-x/2.0)*sp_spec.eval_genlaguerre(n, 2, x), nRange))).T
        return evaluated_basis @ self.solutionWithDirichletBC
    def evaluateSolutionAtPoints(self, x):

        """

        """
        x = np.atleast_1d(x)
        nRange = np.arange(start=0, stop=self.__approxOrder, step=1)
        evaluated_basis = np.asarray(list(map(lambda n: sp_spec.eval_genlaguerre(n, 2, x), nRange))).T
        return evaluated_basis @ self.solutionWithDirichletBC

    def evaluateMultipleSolutionsAtPoints(self, x):
        x = np.atleast_1d(x)
        nRange = np.arange(start=0, stop=self.__approxOrder, step=1)
        evaluated_basis = np.asarray(list(map(lambda n: sp_spec.eval_genlaguerre(n, 2, x), nRange))).T
        return (evaluated_basis @ self.solutionWithDirichletBC)
