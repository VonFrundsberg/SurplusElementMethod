import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sp_linalg
import FiniteElementMethod.mesh.mesh as MeshClass
import FiniteElementMethod.main as fem
import FiniteElementMethod.element.basicElementUtils as belemUtils
import matplotlib.pyplot as plt
import time as time

def fun(N, a, nn):
    finiteElementObject = fem.FEM()

    gradForm = "integral w(x) grad(u) @ grad(v)"
    boundaryForm1 = "boundaryIntegral w(x) [u] <grad(v) @ n>"
    boundaryForm2 = "boundaryIntegral w(x) [v] <grad(u) @ n>"

    gradForm = lambda trialElement, testElement: belemUtils.integrateBilinearForm1(
        trialElement[0], testElement[0], lambda x: 1, 500)
    def boundaryForm1(trialElement: fem.element.element, elementTest: fem.element.element):
        return belemUtils.evaluateBilinearFormAtBoundary2(
            trialElement=trialElement[0], testElement=elementTest[0], weight=lambda x: 1)
    def boundaryForm2(trialElement: fem.element.element, testElement: fem.element.element):
        return belemUtils.evaluateBilinearFormAtBoundary2(
            trialElement=testElement[0], testElement=trialElement[0], weight=lambda x: 1)

    functional = "integral w(x) u f"
    functional = lambda testElement: belemUtils.integrateFunctional(
        testElement=testElement[0], function=lambda x: np.sin(x), weight=lambda x: 1, integrationPointsAmount=500)

    finiteElementObject.setBilinearForm([gradForm, boundaryForm1, boundaryForm2])
    finiteElementObject.setRHSFunctional(functional)

    boundaryConditions = ['{ "axis": 0, "boundaryPoint": "5.0", "value": 0.0}']

    finiteElementObject.setDirichletBoundaryConditions(boundaryConditions)

    mesh = MeshClass.mesh(1)
    mesh.generateUniformMeshOnRectange([0, a], nn, N)
    mesh.establishNeighbours()

    np.set_printoptions(precision=3, suppress=True)

    mesh.fileWrite("elementsData.txt", "neighboursData.txt")
    mesh.fileRead("elementsData.txt", "neighboursData.txt")
    finiteElementObject.initializeMesh(mesh)
    finiteElementObject.initializeElements()
    finiteElementObject.calculateElements()
    print('done')
    time.sleep(500)


fun(4, np.pi, 2)