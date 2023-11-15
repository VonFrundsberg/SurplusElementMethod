import numpy as np
import GalerkinMethod.mesh.mesh as MeshClass
import GalerkinMethod.Galerkin1d as galerkin
import GalerkinMethod.element.Element1d.element1dUtils as elem1dUtils
import time as time

def fun(N, a, nn):
    galerkinMethodObject = galerkin.GalerkinMethod1d()

    gradForm = "integral w(x) grad(u) @ grad(v)"
    boundaryForm1 = "boundaryIntegral w(x) [u] <grad(v) @ n>"
    boundaryForm2 = "boundaryIntegral w(x) [v] <grad(u) @ n>"

    gradForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm1(
        trialElement, testElement, lambda x: 1, 500)

    def boundaryForm1(trialElement: galerkin.element.Element1d, elementTest: galerkin.element.Element1d):
        return elem1dUtils.evaluateBilinearFormAtBoundary_20(
            trialElement=trialElement, testElement=elementTest, weight=lambda x: 0.5)

    def boundaryForm2(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateBilinearFormAtBoundary_21(
            trialElement=trialElement, testElement=testElement, weight=lambda x: 0.5)

    functional = "integral w(x) u f"
    functional = lambda testElement: elem1dUtils.integrateFunctional(
        testElement=testElement, function=lambda x: np.sin(x), weight=lambda x: 1, integrationPointsAmount=500)

    galerkinMethodObject.setBilinearForm(innerForms=[gradForm], boundaryForms=[boundaryForm1, boundaryForm2])
    galerkinMethodObject.setRHSFunctional([functional])
    boundaryConditions = []
    # boundaryConditions = ['{"boundaryPoint": "np.pi", "boundaryValue": 0.0}',
    #                       '{"boundaryPoint": "0", "boundaryValue": 0.0}']

    galerkinMethodObject.setDirichletBoundaryConditions(boundaryConditions)

    mesh = MeshClass.mesh(1)
    mesh.generateUniformMeshOnRectange([0, a], nn, N)
    mesh.establishNeighbours()

    np.set_printoptions(precision=3, suppress=True)

    mesh.fileWrite("elementsData.txt", "neighboursData.txt")
    mesh.fileRead("elementsData.txt", "neighboursData.txt")
    galerkinMethodObject.initializeMesh(mesh)
    galerkinMethodObject.initializeElements()
    galerkinMethodObject.calculateElements()
    print('done')
    time.sleep(500)


fun(4, np.pi, 2)