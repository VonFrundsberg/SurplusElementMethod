import numpy as np
import GalerkinMethod.mesh.mesh as MeshClass
import GalerkinMethod.Galerkin1d as galerkin
import GalerkinMethod.element.Element1d.element1dUtils as elem1dUtils
import time as time
from mathematics import integrate as integr
import matplotlib.pyplot as plt
import mathematics.spectral as spec


def generalKohnShamRoutine(nucleusCharge: int , initialDensity, approximationOrder, integrationPointsAmount = 500):

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
        file.write("0.0 10.0 " + str(approximationOrder) + " 0.0")
        file.close()
        galerkinSchrodingerMesh.fileRead("elementsDataSchrodinger.txt",
                                         "neighboursDataSchrodinger.txt")
    def poissonOperator():
        gradForm = "integral x * x grad(u) @ grad(v)"
        gradForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm1(
            trialElement, testElement, lambda x: x * x, integrationPointsAmount)
        return gradForm
    def poissonFunctional(density):
        functional = "- 4 * pi * integral x * x * density f"
        return lambda testElement: elem1dUtils.integrateFunctional(
        testElement=testElement,
        function=lambda x: -4 * np.pi * density(x), weight=lambda x: x * x,
        integrationPointsAmount=integrationPointsAmount)

    galerkinSchrodinger = galerkin.GalerkinMethod1d("EIG")
    def staticSchrodingerOperatorPart():
        kineticTerm = "0.5 * integral x**2 grad(u) @ grad(v)"
        kineticTerm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm1(
            trialElement, testElement, lambda x: 0.5 * x * x, integrationPointsAmount)

        V_externalTerm = "-integral nucleusCharge * x * u * v"
        V_externalTerm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm0(
            trialElement, testElement, lambda x: -nucleusCharge * x, integrationPointsAmount)

        sphericalHarmonicsTerm = "0.5 * l * (l + 1) * integral u * v"
        l = 0
        sphericalHarmonicsTerm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm0(
            trialElement, testElement, lambda x: 0.5 * l * (l + 1), integrationPointsAmount)
        return kineticTerm, V_externalTerm, sphericalHarmonicsTerm

    PoissonBoundaryConditions = ['{"boundaryPoint": "np.inf", "boundaryValue": 0.0}']
    SchrodingerBoundaryConditions = ['{"boundaryPoint": "10.0", "boundaryValue": 0.0}']

    galerkinPoisson.setBilinearForm(innerForms=[poissonOperator()], boundaryForms=[])
    generatePoissonMesh()
    galerkinPoisson.setDirichletBoundaryConditions(PoissonBoundaryConditions)

    generateSchrodingerMesh()
    galerkinSchrodinger.setDirichletBoundaryConditions(SchrodingerBoundaryConditions)
    constantSchrodingerOperator = staticSchrodingerOperatorPart()

    def variableSchrodingerPart(density):
        galerkinPoisson.setRHSFunctional([poissonFunctional(density=density)])

        galerkinPoisson.initializeMesh(galerkinPoissonMesh)
        galerkinPoisson.initializeElements()
        galerkinPoisson.calculateElements()
        galerkinPoisson.solveSLAE()

        V_HartreeTerm = "integral x * x * poissonSolution * u * v"
        V_HartreeTerm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm0(
            trialElement, testElement, lambda x: x * x * galerkinPoisson.evaluateSolutionAtPoints(x),
            integrationPointsAmount)
        """FROM ROMANOWSKI 2007"""
        r_s = lambda x: (3.0/(4.0 * np.pi * x))**(1.0/3.0)
        c_x = -(3.0/(2.0 * np.pi))**(2.0/3.0)
        potentialV_x = lambda x: c_x/r_s(x)
        V_xTerm = "integral x * x * V_x * u * v"
        V_xTerm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm0(
            trialElement, testElement, lambda x: x * x * potentialV_x(x),
            integrationPointsAmount)

        a = 0.0621814
        b = 3.72744
        c = 12.9352

        x0 = -0.10498
        q = np.sqrt(4.0 * c - b*b)
        f1 = 2.0 * b / q
        sqrt_rs = lambda r_s: np.sqrt(r_s)

        P = lambda r_s, x: r_s + b * x + c
        f2 = -b * x0/(x0 * x0 + b * x0 + c)
        f3 = 2.0 * (b + 2.0 * x0) * f2 / q
        D = lambda r_s, x: r_s/P(r_s, x)
        Y = lambda r_s, x: np.arctan(q/(2.0 * x + b))
        W = lambda r_s, x: (x - x0)**2/P(r_s, x)

        eps_c = lambda r_s, x: a / 2.0 * (np.log(D(r_s, x)) + f1 * Y(r_s, x) + f2 * np.log(W(r_s, x)) + f3*Y(r_s, x))
        potentialV_c = lambda r_s, x: eps_c(r_s, x) - a/6.0 * (c * (x - x0) - b * x * x0)/((x - x0) * P(r_s, x))
        r_dependent_V_c = lambda r: potentialV_c(r_s(r), np.sqrt(r_s(r)))

        V_cTerm = "integral x * x * V_c * u * v"
        V_cTerm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm0(
            trialElement, testElement, lambda x: x * x * r_dependent_V_c(x),
            integrationPointsAmount)

        return V_HartreeTerm, V_xTerm, V_cTerm


    galerkinSchrodinger.setBilinearForm([constantSchrodingerOperator,
                                         variableSchrodingerPart(initialDensity)], [])
    print("done")
    time.sleep(500)







generalKohnShamRoutine(nucleusCharge=2,
                       initialDensity=lambda x: np.exp(-x),
                       approximationOrder=10,
                       integrationPointsAmount=500)
