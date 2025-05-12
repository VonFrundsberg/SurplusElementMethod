import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
import scipy.linalg as sp_linalg
from SurplusElement.GalerkinMethod.element.Element1d import element1d as element
from SurplusElement.GalerkinMethod.element.Element1d.AuxClasses import DirichletBoundaryCondition
from SurplusElement.GalerkinMethod.element.Element1d.AuxClasses import ApproximationSpaceParameter
import json



class GalerkinMethod1d:
    # A: sparse.csr_matrix
    methodType: str
    approximationSpaceParameter: ApproximationSpaceParameter
    dirichletBoundaryConditions: list[DirichletBoundaryCondition]
    def __init__(self, methodType: str):
        self.methodType = methodType
        self.approximationSpaceParameter = ApproximationSpaceParameter()
        self.dirichletBoundaryConditions = []
    def setBilinearForm(self, innerForms, boundaryForms, rhsForms = []):
        self.innerForms = innerForms
        self.boundaryForms = boundaryForms
        self.rhsForms = rhsForms

    def setRHSFunctional(self, functionals):
        self.functionals = functionals

    def setApproximationSpaceParameters(self, parameters):

        approximationSpaceParameter = ApproximationSpaceParameter()
        parsedJsonParameterInfo = json.loads(parameters)

        context = {
            'np': np
        }
        if "shift" in parsedJsonParameterInfo:
            approximationSpaceParameter.shift = eval(parsedJsonParameterInfo["shift"], context)
        if "s" in parsedJsonParameterInfo:
            approximationSpaceParameter.s = eval(parsedJsonParameterInfo["s"], context)

        self.approximationSpaceParameter = approximationSpaceParameter
    def setDirichletBoundaryConditions(self, boundaryConditions):

        self.dirichletBoundaryConditions = []
        for boundaryCondition in boundaryConditions:
            parsedJsonBCinfo = json.loads(boundaryCondition)
            dirichletBoundaryCondition = DirichletBoundaryCondition()

            context = {
                'np': np
            }
            dirichletBoundaryCondition.boundaryPoint = eval(parsedJsonBCinfo["boundaryPoint"], context)
            dirichletBoundaryCondition.boundaryValue = eval(str(parsedJsonBCinfo["boundaryValue"]), context)
            self.dirichletBoundaryConditions.append(dirichletBoundaryCondition)
    def initializeMesh(self, mesh):
        """Set up already made rectangular Mesh, which is an object of SurplusElementMethod/GalerkinMethod/Mesh class

        Arguments:
        Mesh: list of 2 objects [Mesh.elements, Mesh.neighbours]. Elements contain info about domain decomposition and
        order of polynomial approximation, neighbours contain info about neighbouring elements. More information in the
        corresponding class

        Returns:
        None
        """
        self.mesh = mesh

    def initializeElements(self):
        """
        For each element specified in Mesh and dirichletBoundaryConditions,
        creates an object of Element1d Type
        """
        elementsAmount = self.mesh.getElementsAmount()
        self.elements : list[element.Element1d] = [None] * elementsAmount
        for i in range(elementsAmount):
            tmpElementInfo = self.mesh.elements[i][0]
            interval = tmpElementInfo[:2]
            elementBoundaryConditions = []
            for boundaryCondition in self.dirichletBoundaryConditions:
                if boundaryCondition.boundaryPoint == interval[0] or boundaryCondition.boundaryPoint == interval[1]:
                    elementBoundaryConditions.append(boundaryCondition)
            self.elements[i] = (element.Element1d(tmpElementInfo[:2], approxOrder=tmpElementInfo[-2], elementType=tmpElementInfo[-1],
                    parameters=self.approximationSpaceParameter, dirichletBoundaryConditions=self.dirichletBoundaryConditions))

    def calculateElements(self):
        if self.methodType == "LE":
            self.__calculateElements_LE()
        if self.methodType == "EIG":
            self.__calculateElements_EIG()

    def recalculateRHS(self, functionals, flatten=True):
        self.functionals = functionals
        elementsAmount = self.mesh.getElementsAmount()
        self.functionalElements = [None] * elementsAmount
        rhsFunctionalsAmount = len(self.functionals)
        if flatten == True:
            for i in range(elementsAmount):
                self.functionalElements[i] = (self.functionals[0](self.elements[i])).flatten()
                for j in range(1, rhsFunctionalsAmount):
                    self.functionalElements[i] += (self.functionals[j](self.elements[i])).flatten()
        else:
            for i in range(elementsAmount):
                self.functionalElements[i] = (self.functionals[0](self.elements[i]))
                for j in range(1, rhsFunctionalsAmount):
                    self.functionalElements[i] += (self.functionals[j](self.elements[i]))
    def __calculateElements_LE(self):
        """
        For each element in self.elements, calculates its discretized version,
         using previously initialized bilinearForms, and RHS functional
                """
        elementsAmount = self.mesh.getElementsAmount()

        self.matrixElements = [None] * elementsAmount
        for i in range(elementsAmount):
            self.matrixElements[i] = [None] * elementsAmount
        self.functionalElements = [None] * elementsAmount

        innerFormsAmount = len(self.innerForms)
        boundaryFormsAmount = len(self.boundaryForms)
        rhsFunctionalsAmount = len(self.functionals)
        for i in range(elementsAmount):
            self.matrixElements[i][i] = self.innerForms[0](self.elements[i], self.elements[i])
            for j in range(1, innerFormsAmount):
                self.matrixElements[i][i] += self.innerForms[j](self.elements[i], self.elements[i])

            for boundaryFormNumber in range(boundaryFormsAmount):
                self.matrixElements[i][i] += self.boundaryForms[boundaryFormNumber](self.elements[i], self.elements[i])

            self.functionalElements[i] = (self.functionals[0](self.elements[i])).flatten()
            for j in range(1, rhsFunctionalsAmount):
                self.functionalElements[i] += (self.functionals[j](self.elements[i])).flatten()
            if len(self.mesh.neighbours) > 0:
                for neighborNumber in self.mesh.neighbours[i]:
                    if i < neighborNumber:
                        self.matrixElements[i][neighborNumber] = self.boundaryForms[0](self.elements[i], self.elements[neighborNumber])
                        for boundaryFormNumber in range(1, boundaryFormsAmount):
                                self.matrixElements[i][neighborNumber] +=\
                                    self.boundaryForms[boundaryFormNumber](self.elements[i], self.elements[neighborNumber])

                        """in case of non-symmetric operator"""
                        # self.matrixElements[neighborNumber][i] = self.boundaryForms[0](self.elements[neighborNumber],
                        #                                                                    self.elements[i])
                        # for boundaryFormNumber in range(1, boundaryFormsAmount):
                        #         self.matrixElements[neighborNumber][i] += \
                        #             self.boundaryForms[boundaryFormNumber](self.elements[neighborNumber], self.elements[i])
                    else:
                        self.matrixElements[i][neighborNumber] = (self.matrixElements[neighborNumber][i]).T
    def recalculateElements_EIG(self, innerFormIndices, innerForms):
        iter = 0
        for j in innerFormIndices:
            self.lhsMatrixElements[j] = innerForms[iter](self.elements[0], self.elements[0])
            iter += 1

    def calculateElementsDenseEig(self):
        """
        For each element in self.elements, calculates its discretized version,
        using previously initialized bilinearForms, assuming the problem is:
        (bilinearForms + boundaryForms) u = lambda rhsForms u
                        """

        innerFormsAmount = len(self.innerForms)
        rhsFormsAmount = len(self.rhsForms)
        self.lhsMatrixElements = [None] * innerFormsAmount
        self.rhsMatrixElements = [None] * rhsFormsAmount

        for j in range(innerFormsAmount):
            self.lhsMatrixElements[j] = self.innerForms[j](self.elements[0], self.elements[0])
        for j in range(rhsFormsAmount):
            self.rhsMatrixElements[j] = self.rhsForms[j](self.elements[0], self.elements[0])


    def __calculateElements_EIG(self):
        """
        For each element in self.elements, calculates its discretized version,
         using previously initialized bilinearForms, assuming the problem is:
          (bilinearForms + boundaryForms) u = lambda rhsForms u
                """
        elementsAmount = self.mesh.getElementsAmount()

        self.lhsMatrixElements = [None] * elementsAmount
        self.rhsMatrixElements = [None] * elementsAmount
        for i in range(elementsAmount):
            self.lhsMatrixElements[i] = [None] * elementsAmount
            self.rhsMatrixElements[i] = [None] * elementsAmount


        innerFormsAmount = len(self.innerForms)
        boundaryFormsAmount = len(self.boundaryForms)
        rhsFormsAmount = len(self.rhsForms)
        for i in range(elementsAmount):
            self.lhsMatrixElements[i][i] = self.innerForms[0](self.elements[i], self.elements[i])
            self.rhsMatrixElements[i][i] = self.rhsForms[0](self.elements[i], self.elements[i])

            for j in range(1, innerFormsAmount):
                self.lhsMatrixElements[i][i] += self.innerForms[j](self.elements[i], self.elements[i])
            for j in range(1, rhsFormsAmount):
                self.rhsMatrixElements[i][i] += self.rhsForms[j](self.elements[i], self.elements[i])

            for boundaryFormNumber in range(boundaryFormsAmount):
                self.lhsMatrixElements[i][i] += self.boundaryForms[boundaryFormNumber](self.elements[i], self.elements[i])


            if len(self.mesh.neighbours) > 0:
                for neighborNumber in self.mesh.neighbours[i]:
                    if i < neighborNumber:
                        self.lhsMatrixElements[i][neighborNumber] = self.boundaryForms[0](self.elements[i], self.elements[neighborNumber])
                        for boundaryFormNumber in range(1, boundaryFormsAmount):
                                self.lhsMatrixElements[i][neighborNumber] +=\
                                    self.boundaryForms[boundaryFormNumber](self.elements[i], self.elements[neighborNumber])

                        """in case of non-symmetric operator"""
                        # self.matrixElements[neighborNumber][i] = self.boundaryForms[0](self.elements[neighborNumber],
                        #                                                                    self.elements[i])
                        # for boundaryFormNumber in range(1, boundaryFormsAmount):
                        #         self.matrixElements[neighborNumber][i] += \
                        #             self.boundaryForms[boundaryFormNumber](self.elements[neighborNumber], self.elements[i])
                    else:
                        self.lhsMatrixElements[i][neighborNumber] = (self.lhsMatrixElements[neighborNumber][i]).T

    def getAmountOfNonZeroSLAE_elements(self):
        return self.A.count_nonzero()

    def checkPositiveEigenvalues(self):
        A = self.A.toarray()
        eigvals = sp_linalg.eigvals(A)
        realParts = np.real(eigvals)
        if np.min(realParts) >= 0.0:
            print("all eigs are positive")
        else:
            print("some eigs aren't poisitive, amount of negative is: ", np.size(np.where(realParts < 0)))
    def getMatrices(self, sumMatrices=True):
        if sumMatrices:
            A = np.sum(np.asarray(self.lhsMatrixElements), axis=0)
            B = np.sum(np.asarray(self.rhsMatrixElements), axis=0)
        else:
            A = self.lhsMatrixElements
            B = self.rhsMatrixElements

        A = A[~(A == 0).all(1), :][:, ~(A == 0).all(0)]
        B = B[~(B == 0).all(1), :][:, ~(B == 0).all(0)]
        return A, B

    def evaluateBasisAtPoints(self, points, elementNumber:int = 0):
        I = np.eye(self.elements[elementNumber].approxOrder)
        I[0, 0] = 0
        I[-1, -1] = 0
        return self.elements[elementNumber].evaluateExpansion(
            I, points)
    def solveEIG_denseMatrix(self, realize=True, amountOfEigs=1, sumMatrices=True, matricesOutput=False):
        """Only for single-domain case.
           Only for zero Dirichlet or Neumann boundary conditions
            Solves generalized eigenvalue problem:
            lhsMatrixElements @ u = lambda * rhsMatrixElements @ u

            Returns:
                ALL eigenpairs"""
        if sumMatrices:
            A = np.sum(np.asarray(self.lhsMatrixElements), axis=0)
            B = np.sum(np.asarray(self.rhsMatrixElements), axis=0)
        else:
            A = self.lhsMatrixElements
            B = self.rhsMatrixElements
        if matricesOutput:
            np.savetxt(X=A, fname="A.txt", fmt='%1.3f')
            np.savetxt(X=B, fname="B.txt", fmt='%1.3f')
        # print(B)
        ind = ~(A==0).all(1)

        A = A[~(A==0).all(1), :][:, ~(A==0).all(0)]
        B = B[~(B == 0).all(1), :][:, ~(B == 0).all(0)]
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
            self.solutionWithDirichletBC = np.zeros(ind.shape, dtype=float)
            self.solutionWithDirichletBC[ind] = vectors[:, 0]
        else:
            self.solutionWithDirichletBC = np.zeros([*ind.shape, amountOfEigs], dtype=float)
            self.solutionWithDirichletBC[ind, :] = vectors[:, :amountOfEigs]
        return values, vectors
    def getRHS(self, elementIndex:int = 0):
        # print(self.functionalElements[elementIndex].shape)
        return self.functionalElements[elementIndex][self.zeroIndices]
    def invertSLAE(self):
        """
        Inverts SLAE; for small matrices
        """
        A = self.matrixElements[0][0]
        ind = ~(A == 0).all(1)
        self.zeroIndices = ind
        A = A[~(A == 0).all(1), :][:, ~(A == 0).all(0)]
        # self.invertedA = sp_linalg.inv(A)
        self.lu = sp_linalg.lu_factor(A)
    def solveSLAE_dense_invertedMatrix(self):
        b = self.functionalElements[0][self.zeroIndices]
        solution = sp_linalg.lu_solve(*self.lu, b)
        self.solutionWithDirichletBC = np.zeros(self.zeroIndices.shape, dtype=float)
        self.solutionWithDirichletBC[self.zeroIndices] = solution
    def solveSLAE_denseMatrix(self):
        """Only for single-domain case.
                  Only for zero Dirichlet or Neumann boundary conditions

                   Returns:
                       solution of Ax = b"""
        A = self.matrixElements[0][0]
        ind = ~(A == 0).all(1)
        A = A[~(A == 0).all(1), :][:, ~(A == 0).all(0)]
        b = self.functionalElements[0][ind]
        solution = sp_linalg.solve(A, b)
        self.solutionWithDirichletBC = np.zeros(ind.shape, dtype=float)
        self.solutionWithDirichletBC[ind] = solution
    def solveSLAE(self):
        """Solves system matrixElements @ u = functionalElements
            Works only for zero Dirichlet boundary conditions
            Returns:
                solution in the form of one-dimensional array"""
        for i in range(len(self.elements)):
            for j in range(len(self.elements)):
                if self.matrixElements[i][j] is not None:
                    self.matrixElements[i][j] = sparse.csr_matrix(self.matrixElements[i][j])

        A = sparse.bmat(self.matrixElements)
        A = sparse.csr_matrix(A)



        ind = (A.getnnz(1) > 0).copy()
        A = A[A.getnnz(1) > 0, :][:, A.getnnz(0) > 0]
        self.A = A
        self.functionalElements = np.hstack(self.functionalElements)

        self.functionalElements = self.functionalElements[ind]
        self.solutionWithDirichletBC = np.zeros(ind.shape, dtype=float)
        solution = sparse_linalg.spsolve(A, self.functionalElements)
        self.solutionWithDirichletBC[ind] = solution
        return self.solutionWithDirichletBC
    def calculateMeshElementProperties(self):
        """For spherically symmetric Poisson problem on unbounded domain.
        Calculates C_sigma / d(x_i), where x_i represent each boundary,
        except for 0 and inf.
        d(x_i) is defined as
        d(x_i) := 0.5 * (h_{i + 1} + h_i) / (p_{i + 1}^2 + p_i^2).
        C_sigma := 4 * C_G * C_{ab}^a, where C_{ab}^a is defined as in the paper.
        C_G is chosen so that
        d(x_i) <= C_G ()??????????????????????????
        """
        elemInnerProperties = []
        elemBoundaryProperties = []

        for elem in self.elements:
            h = elem.interval[1] - elem.interval[0]
            if h == np.inf:
                h = 1.0
            p = elem.approxOrder
            elemInnerProperties.append(h/p**2)


        for i in range(len(self.elements) - 2):
            hPrev = self.elements[i].interval[1] - self.elements[i].interval[0]
            hNext = self.elements[i + 1].interval[1] - self.elements[i + 1].interval[0]

            pPrev = self.elements[i].approxOrder
            pNext = self.elements[i + 1].approxOrder
            elemBoundaryProperties.append(0.5*(hPrev + hNext)/(pPrev**2 + pNext**2))

        hPrev = self.elements[-2].interval[1] - self.elements[-2].interval[0]
        hNext = 1

        pPrev = self.elements[-2].approxOrder
        pNext = self.elements[-1].approxOrder
        elemBoundaryProperties.append(0.5 * (hPrev + hNext) / (pPrev ** 2 + pNext ** 2))

        elemInnerProperties = np.atleast_1d(elemInnerProperties)
        elemBoundaryProperties = np.atleast_1d(elemBoundaryProperties)

        cT = np.finfo(float).max
        cG = 0.0
        # print(elemInnerProperties, elemBoundaryProperties)

        for i in range(len(self.elements) - 1):
            cLeft = elemBoundaryProperties[i]/elemInnerProperties[i]
            cRight = elemBoundaryProperties[i]/elemInnerProperties[i + 1]
            cMin = min(cLeft, cRight)
            cMax = max(cLeft, cRight)
            # print(elemInnerProperties[i]*cMin, " < ", elemBoundaryProperties[i])
            # print(elemInnerProperties[i + 1] * cMin, " < ", elemBoundaryProperties[i])
            #
            # print(elemBoundaryProperties[i], " < ", elemInnerProperties[i]*cMax)
            # print(elemBoundaryProperties[i], " < ", elemInnerProperties[i + 1]*cMax)
            if cT > cMin:
                cT = cMin
            if cG < cMax:
                cG = cMax

        # for i in range(len(self.elements) - 1):
        #     print(self.elements[i].interval[1])
        #     print(elemInnerProperties[i] * cT, " < ", elemBoundaryProperties[i])
        #     print(elemInnerProperties[i + 1] * cT, " < ", elemBoundaryProperties[i])
        #
        #     print(elemBoundaryProperties[i], " < ", elemInnerProperties[i] * cG)
        #     print(elemBoundaryProperties[i], " < ", elemInnerProperties[i + 1] * cG)
        c_ab = 4 * np.sqrt(5/3)
        c_a = 2 * np.sqrt(10/3)
        C_a_ab = 2 * max(3.0 + c_ab, 2 + c_a)
        C_sigma = 4*cG*C_a_ab
        # print(elemBoundaryProperties)
        elemBoundaryProperties = C_sigma / elemBoundaryProperties
        # self.elemBoundaryProperties = elemBoundaryProperties

        """fix for spherical poisson problem"""
        self.elemBoundaryProperties = np.hstack([0.0, elemBoundaryProperties, 0.0])

        def sigmaDGM_ErrorTerm(x):
            if x == self.elements[0].interval[0]:
                return self.elemBoundaryProperties[0]
            for i in range(len(self.elements)):
                if x == self.elements[i].interval[1]:
                    return self.elemBoundaryProperties[i + 1]
        self.sigmaDGM_ErrorTerm = sigmaDGM_ErrorTerm
        # print(self.elemBoundaryProperties)
    def evaluateSolutionAtPoints(self, x):

        """

        """
        x = np.atleast_1d(x)
        elementsAmount = self.mesh.getElementsAmount()
        evaluatedSolution = np.zeros(x.shape, dtype=float)
        offset = 0
        for elementNumber in range(elementsAmount):
            interval = self.elements[elementNumber].interval
            elementPointIndices = np.squeeze(np.argwhere((x >= interval[0]) & (x <= interval[1])))

            elementCoefficients = self.solutionWithDirichletBC[offset: offset + self.elements[elementNumber].approxOrder]
            offset += self.elements[elementNumber].approxOrder

            evaluatedElementOnLocalGrid = self.elements[elementNumber].evaluateExpansion(
                elementCoefficients, x[elementPointIndices])
            evaluatedSolution[elementPointIndices] = evaluatedElementOnLocalGrid

        return evaluatedSolution
    def evaluateMultipleSolutionsAtPoints(self, x):
        x = np.atleast_1d(x)
        elementsAmount = self.mesh.getElementsAmount()
        # print([*x.shape, functionsArray.shape[1]])
        evaluatedSolution = np.zeros([*x.shape, self.solutionWithDirichletBC.shape[1]], dtype=float)
        offset = 0
        for elementNumber in range(elementsAmount):
            interval = self.elements[elementNumber].interval
            elementPointIndices = np.squeeze(np.argwhere((x >= interval[0]) & (x <= interval[1])))

            elementCoefficients = self.solutionWithDirichletBC[offset: offset + self.elements[elementNumber].approxOrder, :]
            offset += self.elements[elementNumber].approxOrder
            # print(elementCoefficients.shape)
            evaluatedElementOnLocalGrid = self.elements[elementNumber].evaluateExpansion(
                elementCoefficients, x[elementPointIndices])
            # print(evaluatedElementOnLocalGrid.shape)
            evaluatedSolution[elementPointIndices, :] = evaluatedElementOnLocalGrid

        return evaluatedSolution
    def getMeshPoints(self):
        """
        scuffed
        """
        elementsAmount = self.mesh.getElementsAmount()
        grid1d = self.elements[0].getRefNodes()
        for elementNumber in range(1, elementsAmount):
            grid1d = np.append(grid1d, self.elements[elementNumber].getRefNodes())

        return grid1d
    def evaluateFunctionsAtPoints(self, functionsArray, x):

        """

        """
        x = np.atleast_1d(x)
        elementsAmount = self.mesh.getElementsAmount()
        # print([*x.shape, functionsArray.shape[1]])
        evaluatedSolution = np.zeros([*x.shape, functionsArray.shape[1]], dtype=float)
        offset = 0
        for elementNumber in range(elementsAmount):
            interval = self.elements[elementNumber].interval
            elementPointIndices = np.squeeze(np.argwhere((x >= interval[0]) & (x <= interval[1])))

            elementCoefficients = functionsArray[offset: offset + self.elements[elementNumber].approxOrder, :]
            offset += self.elements[elementNumber].approxOrder
            # print(elementCoefficients.shape)
            evaluatedElementOnLocalGrid = self.elements[elementNumber].evaluateExpansion(
                elementCoefficients, x[elementPointIndices])
            # print(evaluatedElementOnLocalGrid.shape)
            evaluatedSolution[elementPointIndices, :] = evaluatedElementOnLocalGrid

        return evaluatedSolution

