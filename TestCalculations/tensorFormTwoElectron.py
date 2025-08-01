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
import scipy.linalg as sp_lin
import SurplusElement.mathematics.approximate as approx
def heliumHF(parameter: float, operatorRank: int, approximationOrder: int, angularL: int,
                  integrationPointsAmount:int, elementType: ElementType, nucleusCharge: int,
                  electronsAmount: int, initialDensity):
    galerkinPoisson = galerkin.GalerkinMethod1d("LE")
    galerkinPoissonMesh = MeshClass.mesh(1)
    galerkinSchrodingerMesh = MeshClass.mesh(1)
    def generatePoissonMesh():
        file = open("elementsDataPoisson.txt", "w")
        file.write("0.0 inf " + str(4 * approximationOrder) + " 1.0")
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
    galerkinPoisson.setApproximationSpaceParameters(parameters)
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

    prevEnergy = 1.0
    energy = 0.0
    iterationNum = 0
    galerkinSchrodinger.setBilinearForm([*constantSchrodingerOperator[:-1]
                                         ], [],
                                        rhsForms=[constantSchrodingerOperator[-1]])
    galerkinSchrodinger.calculateElementsSpectralEig()
    A, B = galerkinSchrodinger.getMatrices()
    curX = np.ones(A.shape[0])
    tensorA = (np.kron(B, A) + np.kron(A, B))
    tensorB = np.kron(B, B)
    tensorCurX = np.random.rand(tensorA.shape[0])
    tensorCurX = np.ones(tensorA.shape[0])
    w, n = integr.reg_22_wn(-1, 1, integrationPointsAmount)
    mappedNodes = galerkinSchrodinger.elements[0].map(n)
    # print("bp", mappedNodes[-1])
    jacobian = galerkinSchrodinger.elements[0].inverseDerivativeMap(n)
    def evaluateALot():

        def evaluatedBasis(x):
            evaluatedBasis = galerkinSchrodinger.evaluateBasisAtPoints(x).T
            # print(evaluatedBasis.shape)
            basisProduct = np.einsum("ik, jk -> ijk", evaluatedBasis, evaluatedBasis)
            # print(basisProduct.shape)
            return basisProduct


        functional = lambda testElement: elem1dUtils.integrateTensorFunctional(
            testElement=testElement,
            function=lambda x: evaluatedBasis(x), weight=lambda x: -4 * np.pi,
            tensorShape=(approximationOrder, approximationOrder),
            integrationPointsAmount=integrationPointsAmount)

        galerkinPoisson.recalculateRHS([functional], flatten=False)
        # np.set_printoptions(precision=3, suppress=True)
        invertedPoisson = galerkinPoisson.invertedA
        rhs = galerkinPoisson.getRHS(0)
        # print(invertedPoisson.shape, rhs.shape)
        Ykl = np.einsum("ij, ikl -> jkl", invertedPoisson, rhs)
        shapeYkl = Ykl.shape
        Ykl = np.reshape(Ykl, [shapeYkl[0], shapeYkl[1] * shapeYkl[2]])
        paddedYkl = np.vstack([Ykl, np.zeros((1, shapeYkl[1] * shapeYkl[2]), dtype=float)])
        # print(paddedYkl.shape)
        paddedYklSchrodBasis = lambda x: galerkinPoisson.evaluateFunctionsAtPoints(paddedYkl, x)
        # paddedYklSchrodBasis = galerkinPoisson.evaluateFunctionsAtPoints(paddedYkl, mappedNodes)
        fx = evaluatedBasis(mappedNodes)
        # print(np.linalg.norm(fx.T - fx))
        # print(fx[3, 5, 150]-fx[5, 3, 150])
        # print(fx[0][0])
        wx = (w * jacobian * paddedYklSchrodBasis(mappedNodes).T)
        # for i in range(2):
        #     for j in range(2):
        #         plt.plot(mappedNodes, fx[i, j, :])
        # plt.show()
        wx = np.reshape(wx, [shapeYkl[1], shapeYkl[2], wx.shape[1]])
        # print(fx.shape, wx.shape)
        # integral = np.einsum("ijn, kln -> ijkl", fx, wx)
        # integral = np.transpose(integral, [0, 2, 1, 3])
        integral = np.einsum("ikn, jln -> ijkl", fx, wx)
        # integral = np.transpose(integral, [0, 2, 1, 3])
        arrLen = int(integral.shape[0])
        integral = np.reshape(integral, [arrLen * arrLen, arrLen * arrLen])
        # print(np.linalg.norm(integral.T - integral))
        zerofyiedIntegral = integral[~(integral == 0).all(1), :][:, ~(integral == 0).all(0)]
        # zerofyiedIntegralTT = approx.matrixTTsvd(zerofyiedIntegral, np.array([18, 18]), tol=1e-6)
        # approx.printTT(zerofyiedIntegralTT)
        # print("is Y symmetric", np.linalg.norm(zerofyiedIntegral.T - zerofyiedIntegral))
        # time.sleep(500)
        # print(print(np.min(np.abs(zerofyiedIntegral))))
        return zerofyiedIntegral

    while np.abs(prevEnergy - energy) > 1e-12 and iterationNum < 1000:
        iterationNum += 1
        prevEnergy = energy
        # galerkinSchrodinger.recalculateElements_EIG([-1], variableSchrodingerPart(density))
        # A, B = galerkinSchrodinger.getMatrices()
        # print("A, B norms: ", np.linalg.norm(A - A.T), np.linalg.norm(B - B.T))
        # curX = GD.YunhoGradientDescent(A, B, curX, alpha=1 * 1e-2, gamma=10, output=False, maxIter=2000)
        # print(A.shape)
        # eigval = (curX @ A @ curX) / (curX @ B @ curX)
        # print("HF eigval", eigval)
        tensorY = evaluateALot()
        interactionTensor = tensorY / (4 * np.pi)
        dimSize = int(np.sqrt(np.size(tensorCurX)))

        # eigvals, eigvecs = sp_lin.eigh(interactionTensor, tensorB)

        # print(approximationOrder, sp_lin.det(interactionTensor), sp_lin.det(tensorB))
        # tensorCurX = GD.YunhoGradientDescentWithMomentum(interactionTensor, tensorB, tensorCurX,
        #                     alpha=5 * 1e-3, gamma=15, beta=0.1, output=False, maxIter=80000)
        # tensorCurX = GD.YunhoScipy(interactionTensor, tensorB, tensorCurX, gamma=15)
        # eigval = (tensorCurX @ (interactionTensor) @ tensorCurX) / (tensorCurX @ tensorB @ tensorCurX)
        # print(eigval)
        # time.sleep(1)
        result = GD.YunhoTensorTrain(interactionTensor, tensorA, tensorB, tensorCurX, gamma=15,
            maxRank=15, solTol=1e-16, operatorTol=1e-16, operatorMaxRank=operatorRank, alpha=1e-2, maxIter=5*10**6, eps=1e-13, output=False)
        # eigval = (tensorCurX @ (interactionTensor) @ tensorCurX) / (tensorCurX @ tensorB @ tensorCurX)
        print(operatorRank, result[1] + 2.8790287)
        # eighEigval = (eigvecs[:, 0] @ (interactionTensor) @ eigvecs[:, 0]) / (eigvecs[:, 0] @ tensorB @ eigvecs[:, 0])
        # print(np.min(eigvals), eighEigval, eigval)
        # reshapedTensor = np.reshape(tensorCurX, [dimSize, dimSize])
        # mesh = galerkinSchrodinger.getMeshPoints()
        # chebNodes = spec.chebNodes(approximationOrder, -1, 1)
        # mappedNodes = galerkinSchrodinger.elements[0].map(chebNodes)[1:-1]
        # f = lambda x, y: np.exp(-1.849*(x + y)) * (1 + 0.364 * np.sqrt(x**2 + y**2))
        # xx, yy = np.meshgrid(mappedNodes, mappedNodes)
        # fx = f(xx, yy)
        # spec.barycentricChebInterpolate()
        # plt.plot(fx)
        # plt.show()
        # print(reshapedTensor.shape)
        # reshapedTensor = (((reshapedTensor / mappedNodes).T)/mappedNodes).T
        # plt.plot(mappedNodes, reshapedTensor)
        # plt.show()
        # plt.imshow(reshapedTensor)
        # plt.show()
        # u, s, v = sp_lin.svd(reshapedTensor, full_matrices=False)
        # print(s)
        # print("tensor eigval", eigval)
        # time.sleep(500)
        # return eigval
        # galerkinSchrodinger.solutionWithDirichletBC = np.hstack([0, curX, 0])

        # print(curX.shape)
        # eigvals, eigvecs = galerkinSchrodinger.solveEIG_denseMatrix()

        # time.sleep(50)
        # try:
        #     eigvals, eigvecs = galerkinSchrodinger.solveEIG_denseMatrix()
        # except:
        #     return np.inf
        # print(eigvals[0], (curX @ A @ curX) / (curX @ B @ curX))
        # w, n = integr.reg_32_wn(-1, 1, integrationPointsAmount)
        # mappedNodes = galerkinSchrodinger.elements[0].map(n)
        # jacobian = galerkinSchrodinger.elements[0].inverseDerivativeMap(n)
        # w = w * jacobian
        # # plt.plot(mappedNodes, jacobian * galerkinSchrodinger.evaluateSolutionAtPoints(mappedNodes) ** 2)
        # # plt.show()
        # normalizationConstant = 4.0 * np.pi * (galerkinSchrodinger.evaluateSolutionAtPoints(mappedNodes) ** 2) @ w
        # density = lambda x: (galerkinSchrodinger.evaluateSolutionAtPoints(x) ** 2
        #                      / normalizationConstant)
        # hartree_energy = 4.0 * np.pi * (density(mappedNodes) * galerkinPoisson.evaluateSolutionAtPoints(mappedNodes)) @ w
        # energy = 2 * eigval - hartree_energy
        # print(energy)
        # energy = 2 * eigvals[0] - hartree_energy
    # return energy
import warnings
warnings.filterwarnings("ignore")

# print("EXP")
# for i in [5, 10, 15]:
#     print("approx order = ", i)
#     for j in [2, 4, 8, 16, 32, 64]:
#         result = heliumHF(parameter=3, operatorRank = j, approximationOrder=i + 2, angularL=0, integrationPointsAmount=10000, elementType=ElementType.EXPONENTIAL_INF_HALF_SPACE,
#                                      nucleusCharge=2, electronsAmount=2, initialDensity=lambda x: 0)
print("LOG")
for i in [5, 10, 15]:
        print("approx order = ", i)
        for j in [2, 4, 8, 16, 32, 64]:
            result = heliumHF(parameter=3, operatorRank=j, approximationOrder=i + 2, angularL=0,
                              integrationPointsAmount=10000, elementType=ElementType.LOGARITHMIC_INF_HALF_SPACE,
                              nucleusCharge=2, electronsAmount=2, initialDensity=lambda x: 0)
    # print(i, result + 2.879028764)
