import numpy as np
import SurplusElement.GalerkinMethod.Mesh.mesh as MeshClass
import SurplusElement.GalerkinMethod.Galerkin1d as galerkin
import SurplusElement.GalerkinMethod.element.Element1d.element1dUtils as elem1dUtils
import time as time
from SurplusElement.mathematics import integrate as integr
import matplotlib.pyplot as plt
import SurplusElement.mathematics.spectral as spec
from SurplusElement.GalerkinMethod.element.Element1d.element1d import ElementType
import scipy.linalg as sp_lin
def heliumHF(parameter: float, approximationOrder: int, angularL: int,
                  integrationPointsAmount:int, elementType: ElementType, nucleusCharge: int,
                  electronsAmount: int, initialDensity):
    galerkinPoisson = galerkin.GalerkinMethod1d("LE")
    galerkinPoissonMesh = MeshClass.mesh(1)
    galerkinSchrodingerMesh = MeshClass.mesh(1)
    def generatePoissonMesh():
        file = open("elementsDataPoisson.txt", "w")
        file.write("0.0 inf " + str(2 * approximationOrder) + " 1.0")
        file.close()
        galerkinPoissonMesh.fileRead("elementsDataPoisson.txt", "neighboursDataPoisson.txt")

    def generateSchrodingerMesh():
        file = open("elementsDataSchrodinger.txt", "w")
        if elementType == ElementType.LINEAR:
            file.write("0.0 " + str(parameter) + " " + str(approximationOrder) + " " + str(elementType.value))
        else:
            file.write("0.0 " + str(np.inf) + " " + str(approximationOrder) + " " + str(elementType.value))
        file.close()
        galerkinSchrodingerMesh.fileRead("elementsDataSchrodinger.txt",
                                         "neighboursDataSchrodinger.txt")

    def poissonOperator():
        gradForm = "-integral x * x grad(u) @ grad(v)"
        gradForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm1(
            trialElement, testElement, lambda x: -x * x, integrationPointsAmount)
        return gradForm

    def poissonFunctional(densityArg):
        functional = "- 4 * pi * integral density"
        return lambda testElement: elem1dUtils.integrateFunctional(
        testElement=testElement,
        function=lambda x: densityArg(x), weight=lambda x: -4 * np.pi,
        integrationPointsAmount=integrationPointsAmount)

    galerkinSchrodinger = galerkin.GalerkinMethod1d("EIG")
    def staticSchrodingerOperatorPart():
        kineticTerm = "0.5 * integral grad(u) @ grad(v)"
        kineticTerm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm1(
            trialElement, testElement, lambda x: 0.5, integrationPointsAmount)

        V_externalTerm = "-integral nucleusCharge / x * u * v"
        V_externalTerm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm0(
            trialElement, testElement, lambda x: -nucleusCharge / x, integrationPointsAmount)

        rhsMatrixTerm = "integral u * v"
        rhsMatrixTerm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm0(
            trialElement, testElement, lambda x: 1.0, integrationPointsAmount)

        return kineticTerm, V_externalTerm, rhsMatrixTerm

    PoissonBoundaryConditions = ['{"boundaryPoint": "np.inf", "boundaryValue": 0.0}']
    SchrodingerBoundaryConditions = []
    parameters = []
    if not elementType in [ElementType.SHIFTED_HERMITE, ElementType.SHIFTED_LAGUERRE]:
        if elementType is ElementType.LINEAR:
            SchrodingerBoundaryConditions = ['{"boundaryPoint": "' + str(0.0) + '", "boundaryValue": 0.0}',
                                             '{"boundaryPoint": "' + str(parameter) + '", "boundaryValue": 0.0}']
        else:
            SchrodingerBoundaryConditions = ['{"boundaryPoint": "' + str(0.0) + '", "boundaryValue": 0.0}',
                                             '{"boundaryPoint": "' + "np.inf" + '", "boundaryValue": 0.0}']
        parameters = '{"s": "' + str(parameter) + '"}'
    else:
        if elementType is ElementType.SHIFTED_HERMITE:
            parameters = '{"shift": "' + str(0.0) + '", "s": "' + str(parameter) + '"}'
        elif elementType is ElementType.SHIFTED_LAGUERRE:
            parameters = '{"shift": "' + str(1.0) + '", "s": "' + str(parameter) + '"}'


    galerkinPoisson.setBilinearForm(innerForms=[poissonOperator()], boundaryForms=[])
    generatePoissonMesh()
    galerkinPoisson.setDirichletBoundaryConditions(PoissonBoundaryConditions)
    galerkinPoisson.setRHSFunctional([poissonFunctional(densityArg=initialDensity)])
    galerkinPoisson.initializeMesh(galerkinPoissonMesh)
    galerkinPoisson.initializeElements()
    galerkinPoisson.calculateElements()
    galerkinPoisson.invertSLAE()


    generateSchrodingerMesh()
    galerkinSchrodinger.setApproximationSpaceParameters(parameters)
    galerkinSchrodinger.setDirichletBoundaryConditions(SchrodingerBoundaryConditions)
    galerkinSchrodinger.initializeMesh(galerkinSchrodingerMesh)
    galerkinSchrodinger.initializeElements()

    w, n = integr.reg_32_wn(-1, 1, integrationPointsAmount)
    mappedNodes = galerkinPoisson.elements[0].map(n)
    jacobian = galerkinSchrodinger.elements[0].inverseDerivativeMap(n)

    def evaluatedBasis(x):
        evaluatedBasis = galerkinSchrodinger.evaluateBasisAtPoints(x).T
        basisProduct = np.einsum("ik, jk -> ijk", evaluatedBasis, evaluatedBasis)
        return basisProduct

    def evaluatedBasisIndex(x, i, j):
        evaluatedBasis = galerkinSchrodinger.evaluateBasisAtPoints(x).T
        basisProduct = np.einsum("ik, jk -> ijk", evaluatedBasis, evaluatedBasis)[i, j, :]
        return basisProduct

    functional = lambda testElement: elem1dUtils.integrateTensorFunctional(
        testElement=testElement,
        function=lambda x: evaluatedBasis(x), weight=lambda x: -4 * np.pi,
        tensorShape=(approximationOrder, approximationOrder),
        integrationPointsAmount=integrationPointsAmount)
    galerkinPoisson.recalculateRHS([functional], flatten=False)
    np.set_printoptions(precision=3, suppress=True)
    invertedPoisson = galerkinPoisson.invertedA
    rhs = galerkinPoisson.getRHS(0)
    print(invertedPoisson.shape, rhs.shape)
    Ykl = np.einsum("ij, ikl -> jkl", invertedPoisson, rhs)
    print(Ykl[:, 0, 0])

    # functional = lambda testElement: elem1dUtils.integrateFunctional(
    #     testElement=testElement,
    #     function=lambda x: evaluatedBasisIndex(x, 0, 0), weight=lambda x: -4 * np.pi,
    #     integrationPointsAmount=integrationPointsAmount)
    # galerkinPoisson.recalculateRHS([functional], flatten=False)
    #
    # invertedPoisson = galerkinPoisson.invertedA
    # rhs = galerkinPoisson.getRHS(0)
    # print(invertedPoisson.shape, rhs.shape)
    # Ykl = invertedPoisson @ rhs
    # print(Ykl)

    # print(Ykl.shape)
    # Ykl = np.reshape(Ykl, [5, 9])
    # print(Ykl.shape)
    # u, s, v = sp_lin.svd(Ykl, full_matrices=False)
    # print(s)
    # print(galerkinPoisson.getRHS(0))
    # print(invertedPoisson.shape)
    # Ykl = np.einsum("invertedPoisson

    return 0


result = heliumHF(parameter=1.42776782052568, approximationOrder=3, angularL=0, integrationPointsAmount=2000, elementType=ElementType.LOGARITHMIC_INF_HALF_SPACE,
                                 nucleusCharge=2, electronsAmount=2, initialDensity=lambda x: 0)
print(result + 2.861679995612)