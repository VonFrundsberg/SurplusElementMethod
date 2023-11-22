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
def fun(meshArg, approximationOrder, mappingType, integrationPointsAmount = 500):
    galerkinMethodObject = galerkin.GalerkinMethod1d()

    gradForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm1(
        trialElement, testElement, lambda x: x * x, integrationPointsAmount)

    def boundaryForm1(trialElement: galerkin.element.Element1d, elementTest: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_JumpComponentMain(
            trialElement=trialElement, testElement=elementTest, weight=lambda x: -x * x)

    def boundaryForm2(trialElement: galerkin.element.Element1d, testElement: galerkin.element.Element1d):
        return elem1dUtils.evaluateDG_JumpComponentSymmetry(
            trialElement=trialElement, testElement=testElement, weight=lambda x: -x * x)

    def dimensionless_BPL_function(x, gamma: float, beta: float):
        lessThanOneArgs = np.where(x <= 1)
        moreThanOneArgs = np.where(x >= 1)
        result = np.zeros(x.shape)
        result[lessThanOneArgs] = x[lessThanOneArgs]**(-gamma)
        result[moreThanOneArgs] = x[moreThanOneArgs]**(-beta)
        # plt.plot(x, result)
        # plt.show()
        return result
    fromJacobian = 2
    gamma = 2 - fromJacobian
    beta = 3 - fromJacobian
    functional = lambda testElement: elem1dUtils.integrateFunctional(
        testElement=testElement, function=lambda x: dimensionless_BPL_function(x, gamma, beta), weight=lambda x: 1, integrationPointsAmount=integrationPointsAmount)

    galerkinMethodObject.setBilinearForm(innerForms=[gradForm], boundaryForms=[boundaryForm1, boundaryForm2])
    galerkinMethodObject.setRHSFunctional([functional])

    boundaryConditions = ['{"boundaryPoint": "np.inf", "boundaryValue": 0.0}']

    galerkinMethodObject.setDirichletBoundaryConditions(boundaryConditions)

    mesh = MeshClass.mesh(1)
    mesh.generateProvidedMeshOnRectange([meshArg],
                                       [approximationOrder], [mappingType])
    mesh.establishNeighbours()

    mesh.fileWrite("elementsData.txt", "neighboursData.txt")
    mesh.fileRead("elementsData.txt", "neighboursData.txt")
    galerkinMethodObject.initializeMesh(mesh)
    galerkinMethodObject.initializeElements()
    galerkinMethodObject.calculateElements()
    galerkinMethodObject.solveSLAE()


    def dimensionless_BPL_asol(x, gamma: float, beta: float):
        result = np.zeros(x.shape)
        lessThanOneArgs = np.where(x <= 1)
        moreThanOneArgs = np.where(x >= 1)

        if gamma == 2.0:
            result[lessThanOneArgs] = (beta - 1.0)/(beta - 2.0) - np.log(x[lessThanOneArgs])
        if beta == 3.0:
            result[moreThanOneArgs] = 1/(x[moreThanOneArgs]) * ((4.0 - gamma)/(3.0 - gamma) + np.log(x[moreThanOneArgs]))
        return result


    w, grid = integr.log_16_wn(0.0, mesh.elements[0][0][1], integrationPointsAmount)

    gridSolution = galerkinMethodObject.evaluateSolutionAtPoints(grid)
    a = mesh.elements[0][0][1]

    error = np.sum(w*(2*np.log(grid)*(gridSolution - 2.0) + (gridSolution - 2.0)**2)) + a * (2.0 + (-2.0 + np.log(a))*np.log(a))
    errors = np.array([], dtype=float)
    errors = np.append(errors, error)
    for i in range(1, mesh.elementsAmount - 1):

        w, grid = integr.reg_32_wn(mesh.elements[i][0][0], mesh.elements[i][0][1], integrationPointsAmount)
        calculatedDimensionless_BPL_asol = dimensionless_BPL_asol(grid, 2.0, 3.0)
        gridSolution = galerkinMethodObject.evaluateSolutionAtPoints(grid)
        # print(mesh.elements[i][0][0], mesh.elements[i][0][1], np.sum(w * (calculatedDimensionless_BPL_asol) ** 2))
        error += np.sum(w * (calculatedDimensionless_BPL_asol - gridSolution) ** 2)
        errors = np.append(errors, np.sum(w * (calculatedDimensionless_BPL_asol - gridSolution) ** 2))
    w, grid = integr.reg_22_wn(-1, 1, 3000)

    mappedGrid = galerkinMethodObject.elements[-1].map(grid)
    calculatedDimensionless_BPL_asol = dimensionless_BPL_asol(mappedGrid, 2.0, 3.0)
    gridSolution = galerkinMethodObject.evaluateSolutionAtPoints(mappedGrid)
    # plt.plot(grid, calculatedDimensionless_BPL_asol)
    # plt.plot(grid, gridSolution)
    # plt.show()
    # print(np.sum(w * (calculatedDimensionless_BPL_asol) ** 2))
    # integrable = lambda x: galerkinMethodObject.elements[-1].inverseDerivativeMap(x) * \
    #                        (dimensionless_BPL_asol(galerkinMethodObject.elements[-1].map(x),3, 2))
    # plt.plot(grid, calculatedDimensionless_BPL_asol - gridSolution)
    # plt.plot(grid, gridSolution)
    # plt.show()
    error += np.sum(w * (calculatedDimensionless_BPL_asol - gridSolution) ** 2)
    errors = np.append(errors, np.sum(w * (calculatedDimensionless_BPL_asol - gridSolution) ** 2))
    # plt.plot(grid, w * galerkinMethodObject.elements[-1].derivativeMap(grid) * (calculatedDimensionless_BPL_asol - gridSolution) ** 2)
    # plt.show()
    # plt.plot(grid, gridSolution - calculatedDimensionless_BPL_asol)
    # plt.show()
    nonZeroAmount = galerkinMethodObject.getAmountOfNonZeroSLAE_elements()
    return nonZeroAmount, error, errors


indices = []
errors = []

amountOfAdditionalIntervalBefore1 = 5
amountOfAdditionalIntervalAfter1 = 5
curApproxOrders = 2*np.ones([amountOfAdditionalIntervalBefore1 + amountOfAdditionalIntervalAfter1], dtype=int)
# curApproxOrders[-1] = 10

# bounds = (amountOfAdditionalInterval + 1) * [(0.0, 1.0)]
RangeBefore1 = np.arange(0, amountOfAdditionalIntervalBefore1)
RangeAfter1 = np.arange(1, amountOfAdditionalIntervalAfter1) + 1
# mesh = np.hstack([0.0, *np.cumsum(x[:amountOfAdditionalInterval]), np.inf])
MeshBefore1 = np.hstack([0.0, (((1 / 4) ** (RangeBefore1))[::-1])])
MeshAfter1 = np.hstack([RangeAfter1 * 1, np.inf])

Mesh = np.hstack([MeshBefore1, MeshAfter1])
print(Mesh)
elemTypes = np.zeros(amountOfAdditionalIntervalBefore1 + amountOfAdditionalIntervalAfter1, dtype=int)
elemTypes[-1] = 1

# totalApproxWeight = np.sum(x)
        # approxOrdersFloat = approxOrdersAmount*np.array([x/totalApproxWeight], dtype=float)
        # approxOrders = np.array(approxOrdersFloat, dtype=int)

for i in range(0, 100, 1):
    indices.append(i)
    def lambdaFun(approxOrders, j):
        tmpApproxOrders = approxOrders.copy()
        tmpApproxOrders[j] += 1
        result = fun(Mesh, np.squeeze(tmpApproxOrders), elemTypes, 2000)
        return result

    argMax = 0
    nonZero, Max, errors = lambdaFun(curApproxOrders, 0)


    for j in range(1, amountOfAdditionalIntervalBefore1 + amountOfAdditionalIntervalAfter1):
        nonZero, error, errors = lambdaFun(curApproxOrders, j)
        if error < Max:
            Max = error
            argMax = j
    curApproxOrders[argMax] += 1
    # print(errors)
    # plt.plot(errors)
    # plt.show()
    print(nonZero, Max, *curApproxOrders)

    # error = sp_opt.direct(func=lambdaFun,
    #                         bounds=bounds, maxiter=10)

    # x = error.get('x')
    # approxOrdersAmount = i
    # totalApproxWeight = np.sum(x)
    # approxOrdersFloat = approxOrdersAmount * np.array([x / totalApproxWeight], dtype=float)
    # approxOrders = np.array(approxOrdersFloat, dtype=int) + 2 * np.ones([amountOfAdditionalInterval + 1], dtype=int)
    # Range = np.arange(0, amountOfAdditionalInterval)
    # # mesh = np.hstack([0.0, *np.cumsum(x[:amountOfAdditionalInterval]), np.inf])
    # Mesh = np.hstack([0.0, (((1/4)**(-1 + Range))[::-1]), np.inf])
    # # print(mesh)
    # elemTypes = np.zeros(amountOfAdditionalInterval + 1, dtype=int)
    # elemTypes[-1] = 1
    #
    # result = fun(Mesh, np.squeeze(approxOrders), elemTypes, 2000)
    # print(i, result)
    # print(np.squeeze(Mesh))
    # print(np.squeeze(approxOrders))
    # errors.append(error)

plt.loglog(indices, errors)
plt.show()