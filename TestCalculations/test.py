import numpy as np
from matplotlib import legend

from SurplusElement.Basic_Galerkin import Galerkin1d as galerkin
import SurplusElement.Basic_Galerkin.element.Element1d.element1dUtils as elem1dUtils
import SurplusElement.Basic_Galerkin.Mesh.mesh as MeshClass
from scipy.optimize import minimize_scalar
from SurplusElement.mathematics import integrate as integr
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy import special as sp_spec
import time as time
"""

Reformulated problem for numerical method is 
-d2u/dx^2 - d(v(x) u(x)) + u(x)/l^2 = g(x)/l^2 on the [0, inf] interval with dirichlet BC set to u(0) = 9, u(inf) = 0
Original problem has been stated as 
-d2(u + 1)/dx^2 + (u(x) + 1)/l^2 = (g(x) + 1)/l^2 on the [0, inf] interval with dirichlet BC set to u(0) = 10, u(inf) = 1
"""

PLI = 10.0
V0 = 200.0
RP = 0.0108
DRP = 0.0068
l_V = 10.0
GIM = 2.7*1e+4
RPDRP = 0.015
X_STAR = 0.02
V0_I = 200.0
a = 1e-3
b = 1.0

def v_I(x):
    x = np.atleast_1d(x)
    result = np.zeros(x.shape)
    lessThanXSTAR = np.where(x < X_STAR)
    moreThanXSTAR = np.where(x >= X_STAR)
    result[lessThanXSTAR] = V0_I
    result[moreThanXSTAR] = V0_I * np.exp(-(x[moreThanXSTAR] - X_STAR) ** 2 / (2.0 * RPDRP ** 2))
    return result
def dv_I(x):
    x = np.atleast_1d(x)
    result = np.zeros(x.shape)
    lessThanXSTAR = np.where(x < X_STAR)
    moreThanXSTAR = np.where(x >= X_STAR)
    result[lessThanXSTAR] = 0.0
    result[moreThanXSTAR] = V0_I * (X_STAR - x[moreThanXSTAR]) * np.exp(-(x[moreThanXSTAR] - X_STAR) ** 2 / (2.0 * RPDRP ** 2)) / RPDRP**2
    return result
# def v_I(x):
#     x = np.atleast_1d(x)
#     result = np.ones(x.shape)
#     return result
# def dv_I(x):
#     x = np.atleast_1d(x)
#     result = np.ones(x.shape)
#     return result * 0

def dv_IExp(x):
    x = np.atleast_1d(x)
    result = np.zeros(x.shape)
    lessThanXSTAR = np.where(x < X_STAR)
    moreThanXSTAR = np.where(x >= X_STAR)
    result[lessThanXSTAR] = V0_I * (b - a) * np.exp(-x[lessThanXSTAR])
    result[moreThanXSTAR] = (b - a) * V0_I * np.exp(-x[moreThanXSTAR] - (x[moreThanXSTAR] - X_STAR) ** 2 / (2.0 * RPDRP ** 2))
    result[moreThanXSTAR] += (X_STAR - x[moreThanXSTAR]) * V0_I * np.exp(-(x[moreThanXSTAR] - X_STAR) ** 2 / (2.0 * RPDRP ** 2)) * (b + (a - b)*np.exp(-x[moreThanXSTAR]))/RPDRP**2
    return result
def g_I(x):
    # X_STAR = 0.0
    x = np.atleast_1d(x)
    result = np.zeros(x.shape)
    lessThanXSTAR = np.where(x < X_STAR)
    moreThanXSTAR = np.where(x >= X_STAR)
    result[lessThanXSTAR] = GIM
    result[moreThanXSTAR] = GIM * np.exp(-(x[moreThanXSTAR] - X_STAR) ** 2 / (2.0 * RPDRP ** 2))
    return result
def spectralGalerkinZero(approximationOrder:int = 50,
                     parameter: float = 1.0,
                         amountOfElements: int = 1,
                     integrationPointsAmount = 500):
    domainSize = np.inf
    galerkinMethodObject = galerkin.GalerkinMethod1d("LS")

    infElementBoundary = 10.0
    gradForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm1(
        trialElement, testElement, lambda x: np.nan_to_num(x=x * 0.0, nan=0.0) + 1.0, integrationPointsAmount)

    fluxForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm2(
        trialElement, testElement, lambda x: np.nan_to_num(x=x * 0.0, nan=0.0) + v_I(x), integrationPointsAmount)

    innerForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm0(
        trialElement, testElement, lambda x: np.nan_to_num(x=x * 0.0, nan=0.0) + 1.0/l_V**2, integrationPointsAmount)
    RHS = lambda x: g_I(x)/l_V**2 + (a - b) * np.exp(-x) + \
                dv_IExp(x) - 1.0/l_V**2 * (np.exp(-x)*(a - b))
    functional = lambda testElement: elem1dUtils.integrateFunctional(
        testElement=testElement, function=lambda x: RHS(x),
            weight=lambda x: np.nan_to_num(x=x * 0.0, nan=0.0) + 1.0, integrationPointsAmount=integrationPointsAmount)
    # functional = lambda testElement: elem1dUtils.integrateFunctional(
    #     testElement=testElement, function=lambda x: g_I(x) / l_V ** 2 + 9.0 * np.exp(-x) * (1.0 - 1.0 / l_V ** 2),
    #     weight=lambda x: np.nan_to_num(x=x * 0.0, nan=0.0) + 1.0, integrationPointsAmount=integrationPointsAmount)
    eps = 1e-14
    sigma = 10 * approximationOrder ** 2
    def DGForm1(trialElement: galerkin.element.Element1d, elementTest: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_JumpComponentMain(
            trialElement=trialElement, testElement=elementTest, weight=lambda x: x * 0.0 - 1.0,
            physicalBoundary=np.array([0, domainSize]), eps = eps)
    def DGForm2(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_JumpComponentSymmetry(
            trialElement=trialElement, testElement=testElement, weight=lambda x: x * 0.0 - 1.0,
            physicalBoundary=np.array([0, domainSize]), eps = eps)

    def fluxDG(trialElement: galerkin.element.Element1d, elementTest: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_centralFlux(
            trialElement=trialElement, testElement=elementTest, weight=lambda x: x * 0.0 - v_I(x),
            physicalBoundary=np.array([0, domainSize]), eps = eps)

    # def fluxBoundary(trialElement: galerkin.element.Element1d, elementTest: galerkin.element.Element1d):
    #     return elem1dUtils.evaluateDG_centralFlux(
    #         trialElement=trialElement, testElement=elementTest, weight=lambda x: x * 0.0 - v_I(x),
    #         physicalBoundary=np.array([0, domainSize]), eps = eps)

    def DGForm3(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_ErrorComponent(
            trialElement=trialElement, testElement=testElement, weight=lambda x: x * 0.0 + sigma,
            physicalBoundary=np.array([0, domainSize]), eps = eps)
    galerkinMethodObject.setBilinearForm(innerForms=[gradForm, innerForm, fluxForm],
                                         discontinuityForms=[DGForm1, DGForm2, DGForm3, fluxDG],
                                         boundaryForms=[]
                                         )
    galerkinMethodObject.setRHSFunctional([functional])
    # galerkinMethodObject.setRHSFunctional([functional])
    parameters = '{"s": "' + str(parameter) + '"}'
    galerkinMethodObject.setApproximationSpaceParameters(parameters)
    boundaryConditions = ['{"boundaryPoint": "0.0", "boundaryValue": 0.0}',
                          '{"boundaryPoint": "np.inf", "boundaryValue": 0.0}']

    # boundaryConditions = []
    galerkinMethodObject.setDirichletBoundaryConditions(boundaryConditions)

    mesh = MeshClass.mesh(1)
    # mesh.generateUniformMeshOnRectange([0, infElementBoundary],
    #                                    [amountOfElements],
    #                                    [approximationOrder])
    # mesh.generateSkewedMeshOnRectange([0, infElementBoundary],
    #                                    [amountOfElements],
    #                                    [approximationOrder])
    # mesh.extendRectangleToInf_AlongAxis_OneDirection("right", infElementBoundary, 0, approximationOrder)
    fileElements = open("elementsData.txt", "w")
    fileElements.write("0.0 0.02 " + str(int(approximationOrder)) + " 0.0" + "\n")
    fileElements.write("0.02 0.05 " + str(int(approximationOrder)) + " 0.0" + "\n")
    fileElements.write("0.05 0.1 " + str(int(approximationOrder)) + " 0.0" + "\n")
    fileElements.write("0.1 0.5 " + str(int(approximationOrder)) + " 0.0" + "\n")
    fileElements.write("0.5 2.0 " + str(int(approximationOrder)) + " 0.0" + "\n")
    fileElements.write("2.0 inf " + str(int(approximationOrder)) + " 1.0" + "\n")
    # fileElements.write("0.0 inf " + str(int(approximationOrder)) + " 1.0" + "\n")
    fileNeighbours = open("neighboursData.txt", "w")
    fileNeighbours.write("1 \n ")
    fileNeighbours.write("0 2 \n ")
    fileNeighbours.write("1 3 \n ")
    fileNeighbours.write("2 4 \n ")
    fileNeighbours.write("3 5 \n ")
    fileNeighbours.write("4 \n ")
    fileElements.close()
    fileNeighbours.close()
    # mesh.extendRectangleToInf_AlongAxis_OneDirection("right", 0.0, 0, approximationOrder)
    # mesh.establishNeighbours()

    # mesh.fileWrite("elementsData.txt", "neighboursData.txt")

    mesh.fileRead("elementsData.txt", "neighboursData.txt")
    galerkinMethodObject.initializeMesh(mesh)
    galerkinMethodObject.initializeElements()
    # print("calculation of elements")
    galerkinMethodObject.calculateElements()
    galerkinMethodObject.solveSLAE()
    # galerkinMethodObject.checkPositiveEigenvalues()

    w, grid = integr.reg_22_wn(0.0, 0.2, integrationPointsAmount)
    # w, n = integr.reg_32_wn(-1, 1, integrationPointsAmount)
    # mappedNodes = galerkinMethodObject.elements[0].map(n)
    # jacobian = galerkinMethodObject.elements[0].inverseDerivativeMap(n)
    # w = w * jacobian
    # gridSolution = galerkinMethodObject.evaluateSolution(mappedNodes) + 9.0 * np.exp(-mappedNodes)
    # gridSolution +=

    # plt.plot(grid, galerkinMethodObject.evaluateSolution(grid) + 9.0 * np.exp(-grid) + 1.0)
    # plt.plot(grid, analyticSolution(grid))
    # plt.show()

    # plt.plot(gridSolution)
    # plt.plot(mappedAnalytic)
    # plt.plot(mappedNodes, w * (gridSolution - mappedAnalytic) ** 2)
    # plt.show()
    # print(mappedNodes)
    gridSol = galerkinMethodObject.evaluateSolution(grid)
    d1gridSol = galerkinMethodObject.evaluateSolutionDerivative(grid, 1)
    d2gridSol = galerkinMethodObject.evaluateSolutionDerivative(grid, 2)
    # print(b, a)
    # plt.plot(grid, gridSol)
    # plt.plot(grid, gridSol + b * np.ones(gridSol.size) + np.exp(-grid) * (a - b))
    # plt.show()
    # gridAnalyticSolution = analyticSolution(grid)
    # plt.plot(grid, galerkinMethodObject.evaluateSolution(grid) + 9.0 * np.exp(-grid))
    # plt.plot(grid, gridAnalyticSolution)
    # plt.show()
    rhsX = RHS(grid)
    numericalRHS = -d2gridSol - d1gridSol * v_I(grid) - gridSol * dv_I(grid) + gridSol/l_V**2
    # plt.plot(grid, rhsX, label="RHS")
    # plt.plot(grid, numericalRHS, label="Numerical")
    # # plt.plot(grid, gridSol + 0.0, label="Solution")
    # plt.legend()
    # plt.show()
    # time.sleep(500)
    return np.max(np.abs(numericalRHS - rhsX))/np.max(np.abs(rhsX))
    return np.sqrt(w @ (numericalRHS - rhsX)**2)/np.sqrt(w @ rhsX**2)
    # return [np.sqrt(w @ (-d2gridSol + gridSol/l_V**2 - rhsX) ** 2) / np.sqrt(w @ (rhsX) ** 2),
    #         np.sqrt(w @ (gridSolution - mappedAnalytic) ** 2) / np.sqrt(w @ (mappedAnalytic) ** 2)]
    return 1.0
    # return np.sqrt(w @ (gridSolution - mappedAnalytic) ** 2) / np.sqrt(w @ (mappedAnalytic) ** 2)
# for i in range(3, 51):
#     bounds = [(0.1, 10.0), (1e-5, 1e+3)]
#     opt.differential_evolution(
#         func=(lambda x: spectralGalerkin(
#         approximationOrder=i, sigmaExp=x[0], parameter=x[1],
#             integrationPointsAmount=10000)),
#         bounds=bounds)

# spectralGalerkin(3, 5.0, 1.0, integrationPointsAmount=100000)

for i in range(3, 21):
    res =spectralGalerkinZero(approximationOrder=i, amountOfElements=10,
                                     parameter=1e+0, integrationPointsAmount=int(1e+4))
    print(i, res)

# for i in range(3, 101):
#     bounds = [(1e-5, 1e+1)]
#     res = opt.basinhopping(
#         func=(lambda x: spectralGalerkinZero(
#         approximationOrder=i, parameter=x[0],
#             integrationPointsAmount=int(1e+5))),
#         x0=[1.0])
#     print(i, *res.get("x"), res.get("fun"))
