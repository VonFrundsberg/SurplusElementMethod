import SurplusElement.mathematics.spectral as spec
import SurplusElement.GalerkinMethod.Galerkin1d as galerkin
import SurplusElement.GalerkinMethod.Mesh.mesh as MeshClass
import SurplusElement.GalerkinMethod.element.Element1d.element1dUtils as elem1dUtils
from SurplusElement.mathematics import integrate as integr

import numpy as np
import scipy.linalg as sp_linalg
from fontTools.varLib.mutator import prev
import matplotlib.pyplot as plt
import time as time
import pandas as pd
import scipy.interpolate as interp


"""Main constants"""
"""N := EN from fortran code"""
N = 1e+3
"""n_e := ENI from fortran code"""
ENIO = 3.87298*1e+16
ENIE = 0.605
ENIF = 1.5
EKB = 8.61709715681519*1e-5
TE = 900
TE_K = TE + 273.15
n_e = 1e-12*ENIO*np.exp(-ENIE/(EKB*TE_K))*TE_K**ENIF
K_S = 1.0
beta_F_1 = 4.0
beta_F_2 = 0.1
beta_E_1 = 0.0076
beta_E_2 = 0.0001
D_E_i = 4.0*1e-7
D_F_i = 1.5*1e-7

"""Defects constants"""
PLI = 10.0
V0 = 200.0
RP = 0.0108
DRP = 0.0068
l_V = 10.0
l_I = l_V
GIM = 2.7*1e+4
RPDRP = 0.015
X_STAR = 0.02

alpha_1 = 1.0
ga_1 = 0.0
alpha_2 = 1.0
beta_2 = -1.0
beta_1_I = -1e-3
beta_1_V = -10.0
ga_2 = 0.0
V0_I = 200.0
V0_V = 0.0

def chi(C):
    return (C - N + np.sqrt((C - N)**2 + 4 * n_e**2))/(2 * n_e)

def D_E(chi):
    return D_E_i * (1.0 + beta_E_1 * chi + beta_E_2 * chi**2)/(1.0 + beta_E_1 + beta_E_2)

def D_F(chi):
    return D_F_i * (1.0 + beta_F_1 * chi + beta_F_2 * chi ** 2) / (1.0 + beta_F_1 + beta_F_2)

def D_N(chi, C, C_V, C_I):
    return C * (D_E(chi) * C_V + D_F(chi) * C_I) / np.sqrt((C - N)**2 + 4 * n_e**2)

def d_V(x):
    return x*0.0 + 1.0
def d_I(x):
    return x*0.0 + 1.0

def k_V(x):
    return x*0.0 + 1.0
def k_I(x):
    return x*0.0 + 1.0

def read_data():
    df_c_0 = pd.read_csv('C_T0.dat', sep='\s+', header=None,
                          converters={i: lambda x: float(x.replace('D', 'E'))
                                      for i in range(2)})
    c_0 = df_c_0.values

    df_c_i = pd.read_csv('C_I.dat', sep='\s+', header=None,
                         converters={i: lambda x: float(x.replace('D', 'E'))
                                     for i in range(2)})
    c_i = df_c_i.values

    df_c_v = pd.read_csv('C_V.dat', sep='\s+', header=None,
                         converters={i: lambda x: float(x.replace('D', 'E'))
                                     for i in range(2)})
    c_v = df_c_v.values

    df_c_ph = pd.read_csv('C_PH.dat', sep='\s+', header=None,
                      converters={i: lambda x: float(x.replace('D', 'E'))
                                 for i in range(2)})
    c_ph = df_c_ph.values


    df_ph_0 = pd.read_csv('ph-impl_o.dat', sep='\s+', header=None,
                      converters={i: lambda x: float(x.replace('D', 'E'))
                                 for i in range(2)})
    ph_0 = df_ph_0.values

    df_ph = pd.read_csv('ph-impl.dat', sep='\s+', header=None,
                          converters={i: lambda x: float(x.replace('D', 'E'))
                                      for i in range(2)})
    ph = df_ph.values

    return c_ph, c_0, c_i, c_v, ph_0, ph

defectsC_I = galerkin.GalerkinMethod1d("LS")
defectsC_V = galerkin.GalerkinMethod1d("LS")
defectsMesh = MeshClass.mesh(1)
defectsApproximationOrder: int = 3
integrationPointsAmount: int = 100000
domainSize = 100

def setDefectsMesh():
    fileElements = open("elementsDataDefects.txt", "w")
    fileNeighbours = open("neighboursDataDefects.txt", "w")
    # fileElements.write("0.0 0.07 " + str(defectsApproximationOrder) + " 0.0" + "\n")
    # fileElements.write("0.07 0.15 " + str(defectsApproximationOrder) + " 0.0" + "\n")
    # fileElements.write("0.15 " + str(domainSize) + " " + str(defectsApproximationOrder) + " 1.0" + "\n")
    fileElements.write("0.0 0.1 " + str(defectsApproximationOrder) + " 0.0" + "\n")
    fileElements.write("0.1 " + str(domainSize) + " " + str(defectsApproximationOrder) + " 0.0" + "\n")
    fileElements.close()
    # fileNeighbours.write("1 \n"
    #                      "0 2 \n"
    #                      "1 \n")
    fileNeighbours.write("1 \n"
                         "0 \n")
    fileNeighbours.close()
    defectsMesh.fileRead("elementsDataDefects.txt", "neighboursDataDefects.txt")

setDefectsMesh()
def defectForms(v, g, l):
    innerForm1 = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm1(
        trialElement, testElement, lambda x: np.nan_to_num(x=x * 0.0, nan=0.0) + 1.0, integrationPointsAmount)
    eps = 1e-15
    sigma = 1e+5 * defectsApproximationOrder ** 4 / 1e-2
    def DGForm1(trialElement: galerkin.element.Element1d, elementTest: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_JumpComponentMain(
            trialElement=trialElement, testElement=elementTest,
            weight=lambda x: np.nan_to_num(x=x * 0.0, nan=0.0) - 1.0,
            physicalBoundary=np.array([0, domainSize]), eps=eps)

    def DGForm2(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_JumpComponentSymmetry(
            trialElement=trialElement, testElement=testElement,
            weight=lambda x: np.nan_to_num(x=x * 0.0, nan=0.0) - 1.0,
            physicalBoundary=np.array([0, domainSize]), eps=eps)
    def DGForm3(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_ErrorComponent(
            trialElement=trialElement, testElement=testElement,
            weight=lambda x: np.nan_to_num(x=x * 0.0, nan=0.0) + sigma,
            physicalBoundary=np.array([0, domainSize]), eps=eps)
    def fluxDGForm(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_upwind(
            trialElement=trialElement, testElement=testElement,
            weight=lambda x: np.nan_to_num(x=x * 0.0, nan=0.0) - 1.0,
            physicalBoundary=np.array([0, domainSize]), eps=eps)
    def minusSignFunction(x):
        if x == 0:
            return np.nan_to_num(x=x * 0.0, nan=0.0) - 1.0
        if x == domainSize:
            return np.nan_to_num(x=x * 0.0, nan=0.0) + 1.0
        return np.nan_to_num(x=x * 0.0, nan=0.0)
    def boundaryForm1(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateBilinearFormAtBoundary_21(
            trialElement=trialElement, testElement=testElement, weight=lambda x: -minusSignFunction(x))

    def boundaryForm2(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateBilinearFormAtBoundary_20(
            trialElement=trialElement, testElement=testElement, weight=lambda x: -minusSignFunction(x))

    def boundaryForm3(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateBilinearFormAtBoundary1(
            trialElement=trialElement, testElement=testElement,
            weight=lambda x: np.nan_to_num(x=x * 0.0, nan=0.0) + sigma, B=0.0)

    def boundaryForm4(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateBilinearFormAtBoundary1(
            trialElement=trialElement, testElement=testElement,
            weight=lambda x: np.nan_to_num(x=x * 0.0, nan=0.0) + sigma, B=domainSize)
    def boundaryForm5(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateBilinearFormAtBoundary1(
            trialElement=trialElement, testElement=testElement,
            weight=lambda x: np.nan_to_num(x=x * 0.0, nan=0.0) - 1.0, B=domainSize)
    fluxForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm2(
        trialElement, testElement, weight=lambda x: v(x), integrationPointsAmount=integrationPointsAmount)

    innerForm3 = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm0(
        trialElement, testElement, lambda x: np.nan_to_num(x=x * 0.0, nan=0.0) + 1.0/l**2, integrationPointsAmount)
    # innerForm3 = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm0(
    #     trialElement, testElement, lambda x: x * 0.0 + 1.0, integrationPointsAmount)
    functional = lambda testElement: elem1dUtils.integrateFunctional(
        testElement=testElement, function=lambda x: g(x)/l**2,
        weight=lambda x: np.nan_to_num(x=x * 0.0, nan=0.0) + 1.0,
        integrationPointsAmount=integrationPointsAmount)

    boundaryFunctional0 = lambda testElement: elem1dUtils.evaluateFunctionalAtBoundaries0(
        testElement=testElement, weight=lambda x: minusSignFunction(x),
        leftValue=-10.0, rightValue=-1.0, leftBoundary=0.0, rightBoundary=domainSize)
    def BCfunction(x):
        if x == 0:
            return np.nan_to_num(x=x * 0.0, nan=0.0) + 10.0
        if x == domainSize:
            return np.nan_to_num(x=x * 0.0, nan=0.0) + 1.0
        return np.nan_to_num(x=x * 0.0, nan=0.0)
    boundaryFunctional1 = lambda testElement: elem1dUtils.evaluateFunctionalAtBoundaries1(
        testElement=testElement, weight=lambda x: BCfunction(x)*sigma,
        leftBoundary=0.0, rightBoundary=domainSize)
    def ZeroLeftFunction(x):
        if x == 0:
            return np.nan_to_num(x=x * 0.0, nan=0.0) + 0.0
        if x == domainSize:
            return np.nan_to_num(x=x * 0.0, nan=0.0) + 1.0
        return np.nan_to_num(x=x * 0.0, nan=0.0)
    boundaryFunctional2 = lambda testElement: elem1dUtils.evaluateFunctionalAtBoundaries1(
        testElement=testElement, weight=lambda x: 10.0*ZeroLeftFunction(x),
        leftBoundary=0.0, rightBoundary=domainSize)

    return (innerForm1, fluxForm, innerForm3,
            DGForm1, DGForm2, DGForm3,
            fluxDGForm,
        boundaryForm1, boundaryForm2, boundaryForm3, boundaryForm4, boundaryForm5,
            functional, boundaryFunctional0, boundaryFunctional1, boundaryFunctional2)

# boundaryConditionsC_I = ['{"boundaryPoint": "0.0", "boundaryValue": 1e-3}',
#                       '{"boundaryPoint": "np.inf", "boundaryValue": 1}']
boundaryConditionsC_V = ['{"boundaryPoint": "0.0", "boundaryValue": 1.0}',
                      '{"boundaryPoint": "' + str(float(domainSize)) + '", "boundaryValue": 1.0}']
# boundaryConditionsC_V = ['{"boundaryPoint": "0.0", "boundaryValue": 1.0}',
#                       '{"boundaryPoint": "np.inf", "boundaryValue": 1.0}']

def defectsCalculation():
    """Vacancy concentration calculations"""
    # v_V = V0_V * np.exp(-(defectNodes - RP) ** 2 / (2 * DRP ** 2))

    # g_V = 1.0 + GIM * np.exp(-(defectNodes - RP) ** 2 / (2 * DRP ** 2))
    def v_V(x):
        x = np.atleast_1d(x)
        result = np.zeros(x.shape)
        lessThanXSTAR = np.where(x <= X_STAR)
        moreThanXSTAR = np.where(x >= X_STAR)
        result[lessThanXSTAR] = V0_V
        result[moreThanXSTAR] = V0_V * np.exp(-(x[moreThanXSTAR] - X_STAR) ** 2 / (2.0 * RPDRP ** 2))
        return result
    def g_V(x):
        x = np.atleast_1d(x)
        result = np.zeros(x.shape)
        lessThanXSTAR = np.where(x <= X_STAR)
        moreThanXSTAR = np.where(x >= X_STAR)
        result[lessThanXSTAR] = 1.0 + GIM
        result[moreThanXSTAR] = 1.0 + GIM * np.exp(-(x[moreThanXSTAR] - X_STAR) ** 2 / (2.0 * RPDRP ** 2))
        return result
    C_V_Forms = defectForms(v_V, g_V, l_V)
    np.set_printoptions(precision=2, suppress=True)
    defectsC_V.setBilinearForm(innerForms=[C_V_Forms[0], C_V_Forms[1], C_V_Forms[2]],
                               discontinuityForms=[C_V_Forms[3], C_V_Forms[4], C_V_Forms[5], C_V_Forms[6]],
                               boundaryForms=[C_V_Forms[8], C_V_Forms[9], C_V_Forms[10], C_V_Forms[11], C_V_Forms[7]])
    # defectsC_V.setBilinearForm(innerForms=[C_V_Forms[0], C_V_Forms[5]],
    #                            discontinuityForms=[C_V_Forms[1], C_V_Forms[2]],
    #                                 boundaryForms=[C_V_Forms[6], C_V_Forms[7], C_V_Forms[8], C_V_Forms[9]])
    defectsC_V.setRHSFunctional(functionals=[C_V_Forms[-4], C_V_Forms[-3], C_V_Forms[-2], C_V_Forms[-1]])

    # defectsC_V.setBilinearForm(innerForms=[C_V_Forms[0], C_V_Forms[5]],
    #                            discontinuityForms=[C_V_Forms[1], C_V_Forms[2]],
    #                            boundaryForms=[C_V_Forms[6], C_V_Forms[7], C_V_Forms[8], C_V_Forms[9]])
    # defectsC_V.setRHSFunctional(functionals=[C_V_Forms[-3]])
    defectsC_V.initializeMesh(defectsMesh)
    defectsC_V.setDirichletBoundaryConditions(boundaryConditionsC_V)
    # parameter = 1.0
    # defectsC_V.setApproximationSpaceParameters(
    #     parameters = '{"s": "' + str(parameter) + '"}')
    defectsC_V.initializeElements()
    # plt.plot(defectsC_V.getMeshPoints(), g_V(defectsC_V.getMeshPoints()))
    # plt.show()
    # defectsC_V.recalculateRHS(functionals=[C_V_Forms[6]])
    defectsC_V.calculateElements()
    sol = defectsC_V.solveSLAE()
    # sol = defectsC_V.solveSLAE_Dense()
    # defectsC_V.
    w, grid = integr.reg_22_wn(0.0, 100, integrationPointsAmount)
    gridSol = defectsC_V.evaluateSolutionAtPoints(grid)
    plt.plot(grid, gridSol)
    plt.show()

    # return C_I, C_V

defectsCalculation()