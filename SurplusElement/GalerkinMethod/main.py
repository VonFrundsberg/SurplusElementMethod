import numpy as np
import scipy.sparse as sparse
from SurplusElement.GalerkinMethod.element import mainElementClass as element
import json

class DirichletBoundaryCondition:
    axis: int
    boundaryPoint: float
    boundaryValue: float

class GalerkinMethod:
    def setBilinearForm(self, bilinearFormsList):
        self.innerForms = [bilinearFormsList[0]]
        self.boundaryForms = [bilinearFormsList[1], bilinearFormsList[2]]

        return None

    def setRHSFunctional(self, functional):
        self.functional = functional
        return None
    def setDirichletBoundaryConditions(self, boundaryConditions):

        self.dirichletBoundaryConditions = []
        for boundaryCondition in boundaryConditions:
            parsedJsonBCinfo = json.loads(boundaryCondition)
            dirichletBoundaryCondition = DirichletBoundaryCondition()

            dirichletBoundaryCondition.axis = parsedJsonBCinfo["axis"]
            dirichletBoundaryCondition.boundaryPoint = parsedJsonBCinfo["boundaryPoint"]
            dirichletBoundaryCondition.boundaryValue = parsedJsonBCinfo["boundaryValue"]
            self.dirichletBoundaryConditions.append(dirichletBoundaryCondition)
    def initializeMesh(self, mesh):
        """Set up already made rectangular Mesh, which is an object of SurplusElementMethod/GalerkinMethod/Mesh class

        Arguments:
        Mesh: list of 2 objects [Mesh.elements, Mesh.neighbours]. Elements contain info about domain decomposition and
        order of polynomial approximation, neighbours contain info about neighbouring elements. More information in the
        corresponding class

        Returns:
        Nothing, creates self.Mesh field in FEM class
        """
        self.mesh = mesh

    def initializeElements(self):
        """
        """
        elementsAmount = self.mesh.getElementsAmount()
        self.elements = [None] * elementsAmount
        for i in range(elementsAmount):
            tmpElement = self.mesh.elements[i]
            interval = tmpElement[:, :2]
            for boundaryCondition in self.dirichletBoundaryConditions:
                if boundaryCondition.boundaryPoint == interval[0]:
                    self.elements[i] = element.element(tmpElement[:, :2], polynomialOrder=tmpElement[:, -2],
                                                       mappingType=tmpElement[:, -1])
                elif boundaryCondition.boundaryPoint == interval[1]:
                    self.elements[i] = element.element(tmpElement[:, :2], polynomialOrder=tmpElement[:, -2],
                                                       mappingType=tmpElement[:, -1])
                else:
                    self.elements[i] = element.element(tmpElement[:, :2], polynomialOrder=tmpElement[:, -2],
                                                    mappingType=tmpElement[:, -1])

    def calculateElements(self):
        """
        For each element in self.Mesh, calculates its discretized version,
         using previously initialized bilinearForms, and RHS functional
                """
        elementsAmount = self.mesh.getElementsAmount()

        self.matrixElements = [None] * elementsAmount
        for i in range(elementsAmount):
            self.matrixElements[i] = [None] * elementsAmount
        self.functionalElements = [None] * elementsAmount

        innerFormsAmount = len(self.innerForms)
        boundaryFormsAmount = len(self.boundaryForms)

        for i in range(elementsAmount):
            innerMatrix = self.innerForms[0](self.elements[i], self.elements[i])

            for j in range(1, innerFormsAmount):
                innerMatrix += self.innerForms[0](self.elements[i], self.elements[i])

            self.matrixElements[i][i] = sparse.csr_matrix(innerMatrix)

            self.functionalElements[i] = (self.functional(self.elements[i])).flatten()

            print(str(i) + ' \'s element calculated')

            for neighborNumber in self.mesh.neighbours[i]:
                self.matrixElements[i][i] = self.boundaryForms[0](self.elements[i], self.elements[i])
                print(self.matrixElements[i][i])
                for boundaryFormNumber in range(1, boundaryFormsAmount):
                    self.matrixElements[i][i] += self.boundaryForms[boundaryFormNumber](self.elements[i], self.elements[i])

                if i < neighborNumber:
                        self.matrixElements[i][neighborNumber] = self.boundaryForms[0](self.elements[i], self.elements[neighborNumber])
                        for boundaryFormNumber in range(1, boundaryFormsAmount):
                            self.matrixElements[i][neighborNumber] +=\
                                self.boundaryForms[boundaryFormNumber](self.elements[i], self.elements[neighborNumber])
                else:
                    self.matrixElements[i][neighborNumber] = self.matrixElements[neighborNumber][i].T

    def solveSLAE(self):
        return None

    def solve(self):
        A = sparse.bmat(self.matrixElems)
        A = sparse.csr_matrix(A)
        ind = (A.getnnz(1) > 0).copy()

        A = A[A.getnnz(1) > 0, :][:, A.getnnz(0) > 0]
        self.rhs = np.hstack(self.rhs)

        self.rhs = self.rhs[ind]
