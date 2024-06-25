import numpy as np
import GalerkinMethod.mesh.mesh as MeshClass
import GalerkinMethod.Galerkin1d as galerkin
import GalerkinMethod.element.Element1d.element1dUtils as elem1dUtils
import time as time
from mathematics import integrate as integr
import matplotlib.pyplot as plt
import mathematics.spectral as spec


def generalKohnShamRoutine(nucleusCharge: int , schrodingerBoundaryPoint: float,
                           initialDensity, approximationOrder, integrationPointsAmount = 500):

    galerkinPoisson = galerkin.GalerkinMethod1d("LE")
    galerkinPoissonMesh = MeshClass.mesh(1)
    galerkinSchrodingerMesh = MeshClass.mesh(1)
    def generatePoissonMesh():
        file = open("elementsDataPoisson.txt", "w")
        file.write("0.0 inf " + str(approximationOrder) + " 1.0")
        file.close()
        galerkinPoissonMesh.fileRead("elementsDataPoisson.txt", "neighboursDataPoisson.txt")

    def generateSchrodingerMesh():
        file = open("elementsDataSchrodinger.txt", "w")
        file.write("0.0 " + str(schrodingerBoundaryPoint) + " " + str(approximationOrder) + " 0.0")
        file.close()
        galerkinSchrodingerMesh.fileRead("elementsDataSchrodinger.txt",
                                         "neighboursDataSchrodinger.txt")

    def poissonOperator():
        gradForm = "-integral x * x grad(u) @ grad(v)"
        gradForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm1(
            trialElement, testElement, lambda x: -x * x, integrationPointsAmount)
        return gradForm

    def poissonFunctional(densityArg):
        functional = "- 4 * pi * integral x * x * density f"
        normalizationConstant = 4*np.pi*integr.reg_32(lambda x: x * x * densityArg(x),
                                    a=0, b=schrodingerBoundaryPoint, n=integrationPointsAmount)
        return lambda testElement: elem1dUtils.integrateFunctional(
        testElement=testElement,
        function=lambda x: - 4 * np.pi * densityArg(x)/normalizationConstant, weight=lambda x: x * x,
        integrationPointsAmount=integrationPointsAmount)

    galerkinSchrodinger = galerkin.GalerkinMethod1d("EIG")
    def staticSchrodingerOperatorPart():
        kineticTerm = "0.5 * integral x**2 grad(u) @ grad(v)"
        kineticTerm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm1(
            trialElement, testElement, lambda x: 0.5 * x * x, integrationPointsAmount)

        V_externalTerm = "-integral nucleusCharge * x * u * v"
        V_externalTerm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm0(
            trialElement, testElement, lambda x: -nucleusCharge * x, integrationPointsAmount)

        rhsMatrixTerm = "integral x * x * u * v"
        rhsMatrixTerm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm0(
            trialElement, testElement, lambda x: x * x, integrationPointsAmount)

        return kineticTerm, V_externalTerm, rhsMatrixTerm

    PoissonBoundaryConditions = ['{"boundaryPoint": "np.inf", "boundaryValue": 0.0}']
    SchrodingerBoundaryConditions = ['{"boundaryPoint": "' + str(schrodingerBoundaryPoint) + '", "boundaryValue": 0.0}']

    galerkinPoisson.setBilinearForm(innerForms=[poissonOperator()], boundaryForms=[])
    generatePoissonMesh()
    galerkinPoisson.setDirichletBoundaryConditions(PoissonBoundaryConditions)

    generateSchrodingerMesh()
    galerkinSchrodinger.setDirichletBoundaryConditions(SchrodingerBoundaryConditions)
    galerkinSchrodinger.initializeMesh(galerkinSchrodingerMesh)
    galerkinSchrodinger.initializeElements()
    constantSchrodingerOperator = staticSchrodingerOperatorPart()

    def variableSchrodingerPart(densityArg):
        galerkinPoisson.setRHSFunctional([poissonFunctional(densityArg=densityArg)])

        galerkinPoisson.initializeMesh(galerkinPoissonMesh)
        galerkinPoisson.initializeElements()
        galerkinPoisson.calculateElements()
        galerkinPoisson.solveSLAE()

        V_HartreeTerm = "integral x * x * poissonSolution * u * v"
        V_HartreeTerm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm0(
            trialElement, testElement, lambda x: x * x * galerkinPoisson.evaluateSolutionAtPoints(x),
            integrationPointsAmount)

        V_xTerm = "-(3/pi)**(1/3) * integral x * x * density**(1/3) * u * v"
        normalizationConstant = 4 * np.pi * integr.reg_32(lambda x: (3.0/np.pi)**(1.0/3.0) * x * x * (densityArg(x))**(1.0/3.0),
                                                         a=0, b=schrodingerBoundaryPoint, n=integrationPointsAmount)

        V_xTerm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm0(
            trialElement, testElement,
            lambda x: - 0 * (3.0/np.pi)**(1.0/3.0) * x * x * (densityArg(x))**(1.0/3.0)/normalizationConstant,
            integrationPointsAmount)

        return V_HartreeTerm, V_xTerm
    density = initialDensity
    # print(constantSchrodingerOperator[-1])
    for iterationNumber in range(20):
        galerkinSchrodinger.setBilinearForm([*constantSchrodingerOperator[:-1],
                                             *variableSchrodingerPart(density)], [],
                                            rhsForms=[constantSchrodingerOperator[-1]])
        galerkinSchrodinger.calculateElements()
        eigvals, eigvecs = galerkinSchrodinger.solveEIG_denseMatrix()
        # plt.plot(galerkinPoisson.evaluateSolutionAtPoints(np.linspace(0, 10, 100)))
        # plt.show()
        hartree_energy = integr.reg_32(lambda x: x * x * galerkinSchrodinger.evaluateSolutionAtPoints(x)**2
                                                 * galerkinPoisson.evaluateSolutionAtPoints(x),
                                               a=0, b=schrodingerBoundaryPoint, n=integrationPointsAmount)
        exchange_energy = integr.reg_32(lambda x: - 0 * x * x * galerkinSchrodinger.evaluateSolutionAtPoints(x)**2
                                                  * (3.0/np.pi)**(1.0/3.0) * density(x)**(1.0/3.0),
                                               a=0, b=schrodingerBoundaryPoint, n=integrationPointsAmount)

        normalizationConstant = 4 * np.pi * integr.reg_32(lambda x: x * x * galerkinSchrodinger.evaluateSolutionAtPoints(x)**2,
                                                          a=0, b=schrodingerBoundaryPoint, n=integrationPointsAmount)
        density = lambda x: galerkinSchrodinger.evaluateSolutionAtPoints(x)**2 / normalizationConstant * 2
        # density = lambda x: galerkinSchrodinger.evaluateSolutionAtPoints(x)**2/normalizationConstant
        # plt.plot(eigvecs[:, 0])
        # plt.show()
        points = np.linspace(0, schrodingerBoundaryPoint, 100)
        # plt.plot(galerkinSchrodinger.solutionWithDirichletBC)
        # plt.show()
        # plt.plot(points, d
        # print(eigvals[0], normalizationConstant*hartree_energy, normalizationConstant * exchange_energy)
        # print(2*eigvals[0] + normalizationConstant*(-hartree_energy + 0.5 * exchange_energy))
        print(eigvals[0], hartree_energy, exchange_energy)
        print(2 * eigvals[0] - hartree_energy + 0.5 * exchange_energy)
        # plt.plot(eigvecs[:, :3])
        # plt.show()


    print("done")
    time.sleep(500)






alpha = 1.0
generalKohnShamRoutine(nucleusCharge=2, schrodingerBoundaryPoint = 10.0,
                       initialDensity=lambda x: (2.0*(2.0 * alpha/ np.pi)**(3.0/4.0)*(np.exp(-alpha * x**2)))**2,
                       approximationOrder=100,
                       integrationPointsAmount=1000)
