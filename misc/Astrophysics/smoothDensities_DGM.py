import numpy as np
import GalerkinMethod.mesh.mesh as MeshClass
import GalerkinMethod.Galerkin1d as galerkin
import GalerkinMethod.element.Element1d.element1dUtils as elem1dUtils
import time as time
from mathematics import integrate as integr
import matplotlib.pyplot as plt
import scipy.optimize as sp_opt
import mathematics.spectral as spec
from scipy import integrate as integrate



"""SUPRESS WARNINGS"""
import warnings
warnings.filterwarnings("ignore")
def sphere_asol(x):
    x = np.atleast_1d(x)
    result = np.zeros(x.shape, dtype=float)
    lessThanOneArgs = np.where(x <= 1)
    moreThanOneArgs = np.where(x >= 1)
    result[lessThanOneArgs] = -2.0/3.0*np.pi*(x[lessThanOneArgs]**2 - 3)
    result[moreThanOneArgs] = 4 * np.pi / 3.0 / x[moreThanOneArgs]
    return result

def plummer_asol(x):
    x = np.atleast_1d(x)
    result = 4.0 * np.pi / np.sqrt(1.0 + x**2)
    return result

def NFW_asol(x):
    x = np.atleast_1d(x)
    result = 4 * np.pi * np.log(1 + x)/x
    return result
def sphere_function(x):
        x = np.atleast_1d(x)
        lessThanOneArgs = np.where(x <= 1)
        moreThanOneArgs = np.where(x >= 1)
        result = np.zeros(x.shape)
        result[lessThanOneArgs] = 1
        result[moreThanOneArgs] = 0
        return result
def plummer_function(x):
    x = np.atleast_1d(x)
    result = 3*(1 + x*x)**(-5/2)
    return result

def NFW_function(x):
    x = np.atleast_1d(x)
    return 1.0/(x * (1 + x)**2)

def fun(meshArg, approximationOrder, mappingType, asol, func, integrationPointsAmount = 500):
    galerkinMethodObject = galerkin.GalerkinMethod1d()

    def gradForm(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.integrateBilinearForm1(
        trialElement, testElement, lambda x: x * x, integrationPointsAmount)

    def boundaryForm1(trialElement: galerkin.element.Element1d, elementTest: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_JumpComponentMain(
            trialElement=trialElement, testElement=elementTest, weight=lambda x: -x * x)

    def boundaryForm2(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_JumpComponentSymmetry(
            trialElement=trialElement, testElement=testElement, weight=lambda x: -x * x)



    boundaryConditions = ['{"boundaryPoint": "np.inf", "boundaryValue": 0.0}']

    galerkinMethodObject.setDirichletBoundaryConditions(boundaryConditions)

    mesh = MeshClass.mesh(1)
    mesh.generateProvidedMeshOnRectange([meshArg],
                                       [approximationOrder], [mappingType])
    mesh.establishNeighbours()

    mesh.fileWrite("elementsData.txt", "neighboursData.txt")
    mesh.fileRead("elementsData.txt", "neighboursData.txt")

    functional = lambda testElement: elem1dUtils.integrateFunctional(
        testElement=testElement, function=lambda x: func(x), weight=lambda x: x * x * 4 * np.pi,
        integrationPointsAmount=integrationPointsAmount)

    galerkinMethodObject.initializeMesh(mesh)
    galerkinMethodObject.initializeElements()
    galerkinMethodObject.calculateMeshElementProperties()

    def boundaryForm3(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_ErrorComponent(
            trialElement=trialElement, testElement=testElement,
            weight=lambda x: galerkinMethodObject.sigmaDGM_ErrorTerm(x) * x * x)


    galerkinMethodObject.setBilinearForm(innerForms=[gradForm],
                                         boundaryForms=[boundaryForm1, boundaryForm2, boundaryForm3])
    galerkinMethodObject.setRHSFunctional([functional])

    galerkinMethodObject.calculateElements()

    galerkinMethodObject.solveSLAE()
    # galerkinMethodObject.checkPositiveEigenvalues()
    error = 0.0
    errors = np.array([], dtype=float)
    # relativeErrors = np.array([], dtype=float)
    errors = np.array([], dtype=float)
    w, grid = integr.reg_22_wn(0.0, mesh.elements[0][0][1], integrationPointsAmount)
    gridSolution = galerkinMethodObject.evaluateSolutionAtPoints(grid)
    calculatedDimensionless_BPL_asol = asol(grid)
    local_error = np.sum(w * (calculatedDimensionless_BPL_asol - gridSolution) ** 2)
    error += local_error
    errors = np.append(errors, local_error)


    for i in range(1, mesh.elementsAmount - 1):
        w, grid = integr.reg_22_wn(mesh.elements[i][0][0], mesh.elements[i][0][1], integrationPointsAmount)
        calculatedDimensionless_BPL_asol = asol(grid)
        gridSolution = galerkinMethodObject.evaluateSolutionAtPoints(grid)
        local_error = np.sum(w * (calculatedDimensionless_BPL_asol - gridSolution) ** 2)
        error += local_error
        errors = np.append(errors, local_error)

    w, grid = integr.reg_22_wn(-1.0, 1.0, integrationPointsAmount)
    mappedGrid = galerkinMethodObject.elements[-1].map(grid)
    gridSolution = galerkinMethodObject.evaluateSolutionAtPoints(mappedGrid)
    calculatedDimensionless_BPL_asol = asol(mappedGrid)
    integrable = galerkinMethodObject.elements[-1].inverseDerivativeMap(mappedGrid) * \
                                    (gridSolution - calculatedDimensionless_BPL_asol)**2
    local_error = np.sum(w * integrable)
    error += local_error
    errors = np.append(errors, local_error)

    nonZeroAmount = galerkinMethodObject.getAmountOfNonZeroSLAE_elements()
    return nonZeroAmount, error, errors

def solveWith_GivenMesh_GivenApproxOrders(Mesh, approxOrders, asol, func):
    elemTypes = np.zeros(approxOrders.size, dtype=int)
    elemTypes[-1] = 1
    nonZero, Max, errors = fun(meshArg=Mesh, approximationOrder=np.squeeze(approxOrders), mappingType=elemTypes,
                               asol=asol, func=func, integrationPointsAmount=5000)
    print(nonZero, Max)

def solveWith_MeshOptimization_GivenApproxOrders_DIRECT_hVariant(
        initGrid, approxOrders, asol, func, boundsMultiplier = 2.0):

    init_h = np.diff(initGrid)[:-1]
    bounds_h = list(map(lambda x: (0, x * boundsMultiplier), init_h))
    global maxError
    maxError = np.finfo(float).max

    def costFunc(x):
        global maxError
        if np.min(x) <= 0:
            return np.inf
        Mesh = np.hstack([0.0, *np.cumsum(x), np.inf])
        elemTypes = np.zeros(approxOrders.size, dtype=int)
        elemTypes[-1] = 1
        result = fun(meshArg=Mesh, approximationOrder=np.squeeze(approxOrders), mappingType=elemTypes,
                     asol=asol, func=func, integrationPointsAmount=2000)[1]
        # print(result, Mesh)

        if result < maxError:
            maxError = result
            # showBestError(x)
        return result

    def showBestError(point):
        global maxError
        Mesh = np.hstack([0.0, *np.cumsum(point), np.inf])
        elemTypes = np.zeros(approxOrders.size, dtype=int)
        elemTypes[-1] = 1
        nonZero, Max, errors = fun(meshArg=Mesh, approximationOrder=np.squeeze(approxOrders), mappingType=elemTypes,
                     asol=asol, func=func, integrationPointsAmount=2000)
        print("error: ", Max, "mesh: ", Mesh, "orders: ", approxOrders, "nonZeroAmount: ", nonZero, "errors: ", errors)

        maxError = Max

        # print("approxOrders: ", approxOrders)
        # print("amount of non-zero", nonZero)

    # showBestError(init_h)
    optimizedResult = sp_opt.direct(costFunc, bounds_h)
    showBestError(optimizedResult.get('x'))



def solveWith_MeshOptimization_GivenApproxOrders_DIRECT_gridVariant(
        initGrid, approxOrders, boundsMultiplier = 0.5):

    init_grid = initGrid[1:-1]
    bounds_grid = list(map(lambda x: (max(0, (x * (1 - boundsMultiplier))), x * (1 + boundsMultiplier)), init_grid))
    print(bounds_grid)
    global maxError

    def costFunc(x):
        global maxError
        if np.min(np.diff(x)) <= 0 or np.min(x) < 0:
            return np.inf
        Mesh = np.hstack([0.0, *x, np.inf])
        elemTypes = np.zeros(approxOrders.size, dtype=int)
        elemTypes[-1] = 1
        result = fun(meshArg=Mesh, approximationOrder=np.squeeze(approxOrders), mappingType=elemTypes,
                     gamma=2, beta=3, integrationPointsAmount=2000)[1]
        # print(result, Mesh)

        if result < maxError:
            maxError = result
            showBestError(x)
        return result

    def showBestError(point):
        global maxError
        Mesh = np.hstack([0.0, *point, np.inf])
        elemTypes = np.zeros(approxOrders.size, dtype=int)
        elemTypes[-1] = 1
        nonZero, Max, errors = fun(meshArg=np.hstack(Mesh),
                                   approximationOrder=np.squeeze(approxOrders), mappingType=elemTypes,
                                   gamma=2, beta=3, integrationPointsAmount=2000)
        print("error: ", Max, "mesh: ", Mesh, "orders: ", approxOrders, "nonZeroAmount: ", nonZero, "errors: ", errors)
        maxError = Max

        # print("approxOrders: ", approxOrders)
        # print("amount of non-zero", nonZero)

    showBestError(init_grid)
    optimizedResult = sp_opt.direct(costFunc, bounds_grid)

def solveWith_MeshOptimization_GivenApproxOrders_BASINHOPPING(initGrid, approxOrders):

    init_h = np.diff(initGrid)[:-1]
    global maxError
    def costFunc(x):
        global maxError
        if np.min(x) <= 0:
            return np.inf
        Mesh = np.hstack([0.0, *np.cumsum(x), np.inf])
        elemTypes = np.zeros(approxOrders.size, dtype=int)
        elemTypes[-1] = 1
        result = fun(meshArg=Mesh, approximationOrder=np.squeeze(approxOrders), mappingType=elemTypes,
                                   gamma=2, beta=3, integrationPointsAmount=2000)[1]
        # print(result, Mesh)

        if result < maxError:
            maxError = result
            showBestError(x)
        return result
    def showBestError(point):
        global maxError
        Mesh = np.hstack([0.0, *np.cumsum(point), np.inf])
        elemTypes = np.zeros(approxOrders.size, dtype=int)
        elemTypes[-1] = 1
        nonZero, Max, errors = fun(meshArg=np.hstack(Mesh),
                                   approximationOrder=np.squeeze(approxOrders), mappingType=elemTypes,
                                   gamma=2, beta=3, integrationPointsAmount=2000)
        print("error: ", Max, "mesh: ", Mesh,  "orders: ", approxOrders,"nonZeroAmount: ", nonZero, "errors: ", errors)
        maxError = Max
        # print("approxOrders: ", approxOrders)
        # print("amount of non-zero", nonZero)

    showBestError(init_h)
    optimizedResult = sp_opt.basinhopping(costFunc, init_h)
for i in range(34, 100):
    # Mesh = np.array([0.0, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1.0, 10.0, 100, 1000, np.inf], dtype=float)
    # approxOrders = i*np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=int)
    # approxOrders[0] = 10
    # solveWith_GivenMesh_GivenApproxOrders(Mesh, approxOrders)
    Mesh = np.array([0.0, 20.0, 200.0, np.inf], dtype=float)
    approxOrders = i * np.ones(Mesh.size - 1, dtype=int)
    # stronglyReducedOrder = int(approxOrders[-1]/2)
    # halfReducedOrder = int(approxOrders[-1])
    # approxOrders[-1] = max(stronglyReducedOrder, 2)
    # approxOrders[-2] = max(halfReducedOrder, 2)
    # solveWith_MeshOptimization_GivenApproxOrders_BASINHOPPING(Mesh, approxOrders)
    # solveWith_GivenMesh_GivenApproxOrders(Mesh, approxOrders, NFW_asol, NFW_function)
    solveWith_MeshOptimization_GivenApproxOrders_DIRECT_hVariant(Mesh, approxOrders, NFW_asol, NFW_function)