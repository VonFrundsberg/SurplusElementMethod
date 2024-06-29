import numpy as np
import GalerkinMethod.mesh.mesh as MeshClass
import GalerkinMethod.Galerkin1d as galerkin
import GalerkinMethod.element.Element1d.element1dUtils as elem1dUtils
import time as time
from mathematics import integrate as integr
import matplotlib.pyplot as plt
import mathematics.spectral as spec


def correlationEnergyParameters(polarized=False):
    if polarized:
        A = 0.01555;
        B = -0.0269;
        C = 0.0014;
        D = -0.0108;
        gamma = -0.0843;
        beta1 = 1.3981;
        beta2 = 0.2611
    else:
        A = 0.0311;
        B = -0.048;
        C = 0.002;
        D = -0.0116;
        gamma = -0.1423;
        beta1 = 1.0529;
        beta2 = 0.3334
    return A, B, C, D, gamma, beta1, beta2





def epsilon_correlation(r, polarized=False):
    A, B, C, D, gamma, beta1, beta2 = correlationEnergyParameters(polarized=polarized)
    r = np.atleast_1d(r)
    result = np.zeros(r.shape)
    lessThanOneArgs = np.where(r < 1.0)
    moreThanOneArgs = np.where(r >= 1.0)
    result[moreThanOneArgs] = gamma / (1.0 + beta1 * np.sqrt(r[moreThanOneArgs]) + beta2 * r[moreThanOneArgs])
    result[lessThanOneArgs] = (A * np.log(r[lessThanOneArgs]) + B
                               + C * r[lessThanOneArgs] * np.log(r[lessThanOneArgs]) + D * r[lessThanOneArgs])
    return result


def V_cPotential(r, polarized=False):
    A, B, C, D, gamma, beta1, beta2 = correlationEnergyParameters(polarized=polarized)
    r = np.atleast_1d(r)
    result = np.zeros(r.shape)
    lessThanOneArgs = np.where(r < 1.0)
    moreThanOneArgs = np.where(r >= 1.0)
    result[moreThanOneArgs] = (epsilon_correlation(r[moreThanOneArgs], polarized) *
                               (1.0 + 7.0 / 6.0 * beta1 * np.sqrt(r[moreThanOneArgs]) + beta2 * r[moreThanOneArgs])
                               / (1.0 + beta1 * np.sqrt(r[moreThanOneArgs]) + beta2 * r[moreThanOneArgs]))
    result[lessThanOneArgs] = (A * np.log(r[lessThanOneArgs]) + B - A / 3.0
                               + 2.0 / 3.0 * C * r[lessThanOneArgs] * np.log(r[lessThanOneArgs])
                               + (2.0 * D - C) / 3.0 * r[lessThanOneArgs])
    return result

def generalKohnShamRoutine(nucleusCharge: int, electronsAmount: int, polarized: bool, schrodingerBoundaryPoint: float,
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
        functional = "(- 4 * pi * integral x * x * density f)"
        # normalizationConstant = 4*np.pi*integr.reg_32(lambda x: x * x * densityArg(x),
        #                             a=0, b=schrodingerBoundaryPoint, n=integrationPointsAmount)
        return lambda testElement: elem1dUtils.integrateFunctional(
        testElement=testElement,
        function=lambda x: -4 * np.pi * densityArg(x), weight=lambda x: x * x,
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
        # normalizationConstant = 4 * np.pi * integr.reg_32(lambda x: (3.0/np.pi)**(1.0/3.0) * x * x * (densityArg(x))**(1.0/3.0),
        #                                                  a=0, b=schrodingerBoundaryPoint, n=integrationPointsAmount)

        V_xTerm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm0(
            trialElement, testElement,
            lambda x: - (3.0/np.pi)**(1.0/3.0) * x * x * (densityArg(x))**(1.0/3.0),
            integrationPointsAmount)

        V_cTerm = "integral x * x * V_cPotential[(3/(4 pi))**1/3 * density**(-1/3)] * u * v"
        V_cTerm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm0(
            trialElement, testElement,
            lambda x: x * x *
                      V_cPotential((3.0/(4.0 * np.pi))**(1.0/3.0) * densityArg(x)**(-1.0 / 3.0), polarized=polarized),
            integrationPointsAmount)

        return V_HartreeTerm, V_xTerm, V_cTerm
    density = initialDensity
    # print(constantSchrodingerOperator[-1])
    for iterationNumber in range(20):
        galerkinSchrodinger.setBilinearForm([*constantSchrodingerOperator[:-1],
                                             *variableSchrodingerPart(density)], [],
                                            rhsForms=[constantSchrodingerOperator[-1]])
        galerkinSchrodinger.calculateElements()
        eigvals, eigvecs = galerkinSchrodinger.solveEIG_denseMatrix()
        # w, n = integr.reg_22_wn(a=0, b=schrodingerBoundaryPoint, n=integrationPointsAmount)

        # sol0 = galerkinSchrodinger.evaluateSolutionAtPoints(n)**2
        # sol0Cheb = galerkinSchrodinger.evaluateSolutionAtPoints()
        galerkinSchrodinger.solutionWithDirichletBC[:-1] = eigvecs[:, 1]
        # sol1 = galerkinSchrodinger.evaluateSolutionAtPoints(n)**2
        amountOfEigvecs = 2
        eigvecsWithZeros = np.vstack([eigvecs[:, :amountOfEigvecs],
                                      np.zeros(amountOfEigvecs, dtype=float)])
        eigvecFuncs = lambda x: galerkinSchrodinger.evaluateFunctionsAtPoints(eigvecsWithZeros, x)
        # points = galerkinSchrodinger.getMeshPoints()
        points = spec.chebNodes(300, 0, schrodingerBoundaryPoint)
        # plt.plot(points, np.sum(eigvecFuncs(points)**2, axis=1))
        # plt.show()
        evaluatedEigvecs = np.sum(eigvecFuncs(points)**2, axis=1)
        density = lambda x: spec.barycentricChebInterpolate(
            evaluatedEigvecs, x, a=0, b=schrodingerBoundaryPoint)
        # points = np.linspace(0, schrodingerBoundaryPoint, 1000)
        # plt.plot(points, points * points * density(points) ** (4.0 / 3.0))
        # plt.plot(points, density(points))
        # plt.show()
        #
        normalizationConstant = integr.reg_32(
            lambda x: 4.0 * np.pi * x * x * density(x),
            a=0, b=schrodingerBoundaryPoint, n=integrationPointsAmount)
        density = lambda x: (spec.barycentricChebInterpolate(
            evaluatedEigvecs, x, a=0, b=schrodingerBoundaryPoint)
                             / normalizationConstant) * electronsAmount

        hartree_energy = 4.0 * np.pi * integr.reg_32(lambda x: x * x * density(x)
                                                               * galerkinPoisson.evaluateSolutionAtPoints(x),
                                                     a=0, b=schrodingerBoundaryPoint, n=integrationPointsAmount)
        exchange_energy = (1.0 / 4.0 * 4.0 * np.pi) * (3.0 / np.pi) ** (1.0 / 3.0) * \
                          integr.reg_32(lambda x: x * x * density(x) ** (4.0 / 3.0),
                                        a=0, b=schrodingerBoundaryPoint, n=integrationPointsAmount)
        correlation_energy = 4.0 * np.pi * integr.reg_32(lambda x:
                                                         x * x * density(x) * epsilon_correlation(
                                                             (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) * density(x) ** (
                                                                         -1.0 / 3.0), polarized=polarized),
                                                         a=0, b=schrodingerBoundaryPoint, n=integrationPointsAmount)
        correlation_potential_energy = 4.0 * np.pi * integr.reg_32(lambda x:
                                                                   x * x * density(x) * V_cPotential(
                                                                       (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) * density(
                                                                           x) ** (-1.0 / 3.0), polarized=polarized),
                                                                   a=0, b=schrodingerBoundaryPoint,
                                                                   n=integrationPointsAmount)

        # print('done')
        # time.sleep(500)
        # density = lambda x: galerkinSchrodinger.evaluateSolutionAtPoints(x)**2/normalizationConstant
        # plt.plot(eigvecs[:, 0])
        # plt.show()
        # points = np.linspace(0, schrodingerBoundaryPoint, 100)
        # plt.plot(points, points*points*density(points)**(4.0/3.0))
        # plt.show()
        # plt.plot(points, d
        # print(eigvals[0], normalizationConstant*hartree_energy, normalizationConstant * exchange_energy)
        # print(2*eigvals[0] + normalizationConstant*(-hartree_energy + 0.5 * exchange_energy))
        print([eigvals[0], eigvals[1]], 0.5*hartree_energy, exchange_energy, correlation_energy - correlation_potential_energy)
        print(2*np.sum(eigvals[:2]) - 0.5*hartree_energy + exchange_energy + correlation_energy - correlation_potential_energy)
        # plt.plot(eigvecs[:, :3])
        # plt.show()


    print("done")
    time.sleep(500)






generalKohnShamRoutine(nucleusCharge=4, electronsAmount=4, schrodingerBoundaryPoint = 7.5, polarized=False,
                       initialDensity=lambda x: (2.0*(2.0 * 1.0/ np.pi)**(3.0/4.0)*(np.exp(-1.0 * x**2)))**2,
                       approximationOrder=100,
                       integrationPointsAmount=1000)
