import numpy as np
import SurplusElement.GalerkinMethod.Mesh.mesh as MeshClass
import SurplusElement.GalerkinMethod.Galerkin1d as galerkin
import SurplusElement.GalerkinMethod.element.Element1d.element1dUtils as elem1dUtils
import time as time
from SurplusElement.mathematics import integrate as integr
import matplotlib.pyplot as plt
import SurplusElement.mathematics.spectral as spec
from SurplusElement.GalerkinMethod.element.Element1d.element1d import ElementType
import SurplusElement.optimization.GradientDescentQuadratic as GD
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
    constantSchrodingerOperator = staticSchrodingerOperatorPart()

    def variableSchrodingerPart(densityArg):
        galerkinPoisson.recalculateRHS([poissonFunctional(densityArg=densityArg)])
        print(galerkinPoisson.invertedA.shape)
        galerkinPoisson.solveSLAE_dense_invertedMatrix()

        V_HartreeTerm = "integral poissonSolution * u * v"
        V_HartreeTerm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm0(
            trialElement, testElement, lambda x: galerkinPoisson.evaluateSolutionAtPoints(x),
            integrationPointsAmount)

        return [V_HartreeTerm]
    density = initialDensity
    prevEnergy = 1.0
    energy = 0.0
    iterationNum = 0
    galerkinSchrodinger.setBilinearForm([*constantSchrodingerOperator[:-1],
                                         *variableSchrodingerPart(density)], [],
                                        rhsForms=[constantSchrodingerOperator[-1]])
    galerkinSchrodinger.calculateElementsDenseEig()
    A, B = galerkinSchrodinger.getMatrices()
    curX = np.ones(A.shape[0])
    while np.abs(prevEnergy - energy) > 1e-12 and iterationNum < 1000:
        iterationNum += 1
        prevEnergy = energy
        galerkinSchrodinger.recalculateElements_EIG([-1], variableSchrodingerPart(density))
        A, B = galerkinSchrodinger.getMatrices()
        # print("A, B norms: ", np.linalg.norm(A - A.T), np.linalg.norm(B - B.T))
        curX = GD.YunhoGradientDescent(A, B, curX, alpha=1 * 1e-3, gamma=10, output=False, maxIter=2000)
        # print(A.shape)
        eigval = (curX @ A @ curX) / (curX @ B @ curX)
        galerkinSchrodinger.solutionWithDirichletBC = np.hstack([0, curX, 0])
        # print(curX.shape)
        # eigvals, eigvecs = galerkinSchrodinger.solveEIG_denseMatrix()

        # time.sleep(50)
        # try:
        #     eigvals, eigvecs = galerkinSchrodinger.solveEIG_denseMatrix()
        # except:
        #     return np.inf
        # print(eigvals[0], (curX @ A @ curX) / (curX @ B @ curX))
        w, n = integr.reg_32_wn(-1, 1, integrationPointsAmount)
        mappedNodes = galerkinSchrodinger.elements[0].map(n)
        jacobian = galerkinSchrodinger.elements[0].inverseDerivativeMap(n)
        w = w * jacobian
        # plt.plot(mappedNodes, jacobian * galerkinSchrodinger.evaluateSolutionAtPoints(mappedNodes) ** 2)
        # plt.show()
        normalizationConstant = 4.0 * np.pi * (galerkinSchrodinger.evaluateSolutionAtPoints(mappedNodes) ** 2) @ w
        density = lambda x: (galerkinSchrodinger.evaluateSolutionAtPoints(x) ** 2
                             / normalizationConstant)
        hartree_energy = 4.0 * np.pi * (density(mappedNodes) * galerkinPoisson.evaluateSolutionAtPoints(mappedNodes)) @ w
        energy = 2 * eigval - hartree_energy
        print(energy)
        # energy = 2 * eigvals[0] - hartree_energy
    return energy


result = heliumHF(parameter=1.42776782052568, approximationOrder=5, angularL=0, integrationPointsAmount=2000, elementType=ElementType.LOGARITHMIC_INF_HALF_SPACE,
                                 nucleusCharge=2, electronsAmount=2, initialDensity=lambda x: 0)
print(result + 2.861679995612)