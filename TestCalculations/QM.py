import numpy as np
import SurplusElement.GalerkinMethod.Mesh.mesh as MeshClass
import SurplusElement.GalerkinMethod.Galerkin1d as galerkin
import SurplusElement.GalerkinMethod.element.Element1d.element1dUtils as elem1dUtils
import time as time
from SurplusElement.mathematics import integrate as integr
import matplotlib.pyplot as plt
import SurplusElement.mathematics.spectral as spec
import SurplusElement.optimization.GradientDescentQuadratic as GD

def logarithmicRoutine(parameterS: float, approximationOrder: int,
                       angularL: float, integrationPointsAmount = 500):
    galerkinSchrodingerMesh = MeshClass.mesh(1)
    approximationOrder += 2
    def generateSchrodingerMesh():
        file = open("elementsDataSchrodinger.txt", "w")
        file.write("0.0 " + str(np.inf) + " " + str(approximationOrder) + " 5.0")
        file.close()
        galerkinSchrodingerMesh.fileRead("elementsDataSchrodinger.txt",
                                         "neighboursDataSchrodinger.txt")

    galerkinSchrodinger = galerkin.GalerkinMethod1d("EIG")
    def staticSchrodingerOperatorPart(angularL: float):
        kineticTerm = "0.5 * integral grad(u) @ grad(v)"
        kineticTerm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm1(
            trialElement, testElement, lambda x: 0.5, integrationPointsAmount)

        V_externalTerm = "-1/x * u * v"
        V_externalTerm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm0(
            trialElement, testElement, lambda x: -1.0/x, integrationPointsAmount)

        rhsMatrixTerm = "integral u * v"
        rhsMatrixTerm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm0(
            trialElement, testElement, lambda x: 1.0, integrationPointsAmount)
        sphericalHarmonicsTerm = "0.5 * l * (l + 1) * integral 1.0/x**2 u * v"
        sphericalHarmonicsTerm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm0(
            trialElement, testElement, lambda x: 0.5 * angularL * (angularL + 1.0) / x**2, integrationPointsAmount)
        return kineticTerm, V_externalTerm, sphericalHarmonicsTerm, rhsMatrixTerm

    SchrodingerBoundaryConditions = ['{"boundaryPoint": "' + str(0.0) + '", "boundaryValue": 0.0}',
                                     '{"boundaryPoint": "' + "np.inf" + '", "boundaryValue": 0.0}']
    logarithmicParameters = '{"s": "' + str(parameterS) + '"}'

    generateSchrodingerMesh()
    galerkinSchrodinger.setDirichletBoundaryConditions(SchrodingerBoundaryConditions)
    galerkinSchrodinger.setApproximationSpaceParameters(logarithmicParameters)
    galerkinSchrodinger.initializeMesh(galerkinSchrodingerMesh)

    galerkinSchrodinger.initializeElements()
    schrodingerOperator = staticSchrodingerOperatorPart(angularL)

    galerkinSchrodinger.setBilinearForm(schrodingerOperator[:-1], [],
                                            rhsForms=[schrodingerOperator[-1]])
    galerkinSchrodinger.calculateElementsSpectralEig()
    A, B = galerkinSchrodinger.getMatrices()
    print("A, B norms: ", np.linalg.norm(A - A.T), np.linalg.norm(B - B.T))
    GD.YunhoGradientDescent(A, B, 1, alpha=5*1e-3, gamma=1, output=True, maxIter=10**5)
    eigvals, eigvecs = galerkinSchrodinger.solveEIG_denseMatrix()
    # print(eigvecs)
    # print(eigvals)
    eigvalIndices = np.argsort(np.real(eigvals))
    print(eigvals[0])
    # print(eigvecs.shape)
    # sol = eigvecs[:, 0]
    # print((sol @ A @ sol)/(sol @ B @ sol))

    # return np.real(eigvals[eigvalIndices])

logarithmicRoutine(3, 30, 0, 500)