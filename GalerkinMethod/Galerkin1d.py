import numpy as np
import scipy.sparse as sparse
from GalerkinMethod.element.Element1d import element1d as element
from GalerkinMethod.element.Element1d.DirichletBoundaryCondition import DirichletBoundaryCondition
import json



class GalerkinMethod1d:
    def setBilinearForm(self, innerForms, boundaryForms):
        self.innerForms = innerForms
        self.boundaryForms = boundaryForms

    def setRHSFunctional(self, functionals):
        self.functionals = functionals

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
        """Set up already made rectangular mesh, which is an object of SurplusElementMethod/GalerkinMethod/mesh class

        Arguments:
        mesh: list of 2 objects [mesh.elements, mesh.neighbours]. Elements contain info about domain decomposition and
        order of polynomial approximation, neighbours contain info about neighbouring elements. More information in the
        corresponding class

        Returns:
        Nothing, creates self.mesh field in FEM class
        """
        self.mesh = mesh

    def initializeElements(self):
        """
        """
        elementsAmount = self.mesh.getElementsAmount()
        self.elements = [None] * elementsAmount
        for i in range(elementsAmount):
            tmpElementInfo = self.mesh.elements[i][0]
            interval = tmpElementInfo[:2]
            elementBoundaryConditions = []
            for boundaryCondition in self.dirichletBoundaryConditions:
                if boundaryCondition.boundaryPoint == interval[0] or boundaryCondition.boundaryPoint == interval[1]:
                    elementBoundaryConditions.append(boundaryCondition)

            if len(elementBoundaryConditions) > 0:
                self.elements[i] = element.Element1d(tmpElementInfo[:2], approxOrder=tmpElementInfo[-2],
                                                         elementType=tmpElementInfo[-1],
                                                         dirichletBoundaryConditions=elementBoundaryConditions)
            else:
                 self.elements[i] = element.Element1d(tmpElementInfo[:2], approxOrder=tmpElementInfo[-2],
                                                         elementType=tmpElementInfo[-1])

    def calculateElements(self):
        """
        For each element in self.mesh, calculates its discretized version,
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
            innerMatrix = self.innerForms[0](self.elements[i], self.elements[i])

            for j in range(1, innerFormsAmount):
                innerMatrix += self.innerForms[j](self.elements[i], self.elements[i])

            self.matrixElements[i][i] = innerMatrix

            self.functionalElements[i] = (self.functionals[0](self.elements[i])).flatten()
            for j in range(1, rhsFunctionalsAmount):
                self.functionalElements[i] += (self.functionals[j](self.elements[i])).flatten()

            print(str(i) + ' \'s element calculated')
            print('its grad matrix')
            print(self.matrixElements[i][i])

            for neighborNumber in self.mesh.neighbours[i]:
                for boundaryFormNumber in range(boundaryFormsAmount):
                    print("boundary form ", boundaryFormNumber)
                    print(self.boundaryForms[boundaryFormNumber](self.elements[i], self.elements[i]))
                    self.matrixElements[i][i] += self.boundaryForms[boundaryFormNumber](self.elements[i], self.elements[i])
                if i < neighborNumber:
                        self.matrixElements[i][neighborNumber] = self.boundaryForms[0](self.elements[i], self.elements[neighborNumber])
                        for boundaryFormNumber in range(1, boundaryFormsAmount):
                            self.matrixElements[i][neighborNumber] +=\
                                self.boundaryForms[boundaryFormNumber](self.elements[i], self.elements[neighborNumber])
                else:
                    self.matrixElements[i][neighborNumber] = self.matrixElements[neighborNumber][i].T
            print("resulting matrix")
            print(self.matrixElements[i][i])

    def solveSLAE(self):
        return None

    def solve(self):
        A = sparse.bmat(self.matrixElems)
        A = sparse.csr_matrix(A)
        ind = (A.getnnz(1) > 0).copy()

        A = A[A.getnnz(1) > 0, :][:, A.getnnz(0) > 0]
        self.rhs = np.hstack(self.rhs)

        self.rhs = self.rhs[ind]
