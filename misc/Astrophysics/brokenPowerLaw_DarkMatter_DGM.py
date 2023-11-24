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


def dimensionless_BPL_asol(x, gamma: float, beta: float):
    x = np.atleast_1d(x)
    result = np.zeros(x.shape)
    lessThanOneArgs = np.where(x <= 1)
    moreThanOneArgs = np.where(x >= 1)
    if gamma == 2.0:
        result[lessThanOneArgs] = (beta - 1.0) / (beta - 2.0) - np.log(x[lessThanOneArgs])
    else:
        result[lessThanOneArgs] = 1.0 / (2.0 - gamma)*((beta - gamma) / (beta - 2.0) - ((x[lessThanOneArgs])**(2 - gamma))/(3.0 - gamma))
    if beta == 3.0:
        result[moreThanOneArgs] = 1.0 / (x[moreThanOneArgs]) * (
                    (4.0 - gamma) / (3.0 - gamma) + np.log(x[moreThanOneArgs]))
    else:
        result[moreThanOneArgs] = (1.0 / (beta - 3.0)) / (x[moreThanOneArgs]) * \
                                   ((beta - gamma)/(3.0 - gamma) - ((x[moreThanOneArgs]**(3 - beta))/(beta - 2.0)))
    return result

def dimensionless_BPL_function(x, gamma: float, beta: float):
        lessThanOneArgs = np.where(x <= 1)
        moreThanOneArgs = np.where(x >= 1)
        result = np.zeros(x.shape)
        result[lessThanOneArgs] = x[lessThanOneArgs]**(-gamma)
        result[moreThanOneArgs] = x[moreThanOneArgs]**(-beta)
        return result
def fun(meshArg, approximationOrder, mappingType, gamma: float, beta: float, integrationPointsAmount = 500):
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
    galerkinMethodObject.initializeMesh(mesh)
    galerkinMethodObject.initializeElements()
    galerkinMethodObject.calculateMeshElementProperties()
    fromJacobian = 2

    functional = lambda testElement: elem1dUtils.integrateFunctional(
        testElement=testElement, function=lambda x: dimensionless_BPL_function(x, gamma - fromJacobian, beta - fromJacobian), weight=lambda x: 1,
        integrationPointsAmount=integrationPointsAmount)

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
    # grid = np.linspace(0.0, 5.0, 1000)
    # plt.plot(galerkinMethodObject.evaluateSolutionAtPoints(grid) - dimensionless_BPL_asol(grid, gamma, beta))
    # plt.show()

    # plt.plot(galerkinMethodObject.evaluateSolutionAtPoints(grid))
    # plt.plot(dimensionless_BPL_asol(grid, gamma, beta))
    # plt.show()

    error = 0.0
    errors = np.array([], dtype=float)
    if gamma == 2.0:
        w, grid = integr.log_16_wn(0.0, mesh.elements[0][0][1], integrationPointsAmount)
        gridSolution = galerkinMethodObject.evaluateSolutionAtPoints(grid)
        a = mesh.elements[0][0][1]
        error += np.sum(w*(2*np.log(grid)*(gridSolution - 2.0) + (gridSolution - 2.0)**2)) + a * (2.0 + (-2.0 + np.log(a))*np.log(a))
        errors = np.append(errors, error)
    elif gamma < 2.0:
        w, grid = integr.reg_22_wn(0.0, mesh.elements[0][0][1], integrationPointsAmount)
        gridSolution = galerkinMethodObject.evaluateSolutionAtPoints(grid)
        calculatedDimensionless_BPL_asol = dimensionless_BPL_asol(grid, gamma, beta)
        error += np.sum(w * (calculatedDimensionless_BPL_asol - gridSolution) ** 2)
        errors = np.append(errors, np.sum(w * (calculatedDimensionless_BPL_asol - gridSolution) ** 2))


    for i in range(1, mesh.elementsAmount - 1):
        w, grid = integr.reg_22_wn(mesh.elements[i][0][0], mesh.elements[i][0][1], integrationPointsAmount)
        calculatedDimensionless_BPL_asol = dimensionless_BPL_asol(grid, gamma, beta)
        gridSolution = galerkinMethodObject.evaluateSolutionAtPoints(grid)
        error += np.sum(w * (calculatedDimensionless_BPL_asol - gridSolution) ** 2)
        errors = np.append(errors, np.sum(w * (calculatedDimensionless_BPL_asol - gridSolution) ** 2))

    if beta == 3.0:
        lambdaFunc = lambda x: (dimensionless_BPL_asol(x, 2.0, 3.0) - galerkinMethodObject.evaluateSolutionAtPoints(x))**2
        integral = integrate.quad(func=lambdaFunc, a=galerkinMethodObject.elements[-1].interval[0], b=np.inf, epsabs=1e-16, epsrel=1e-16, limit=100)
        error += integral[0]
        errors = np.append(errors, integral[0])
    else:
        w, grid = integr.reg_22_wn(-1.0, 1.0, integrationPointsAmount)
        mappedGrid = galerkinMethodObject.elements[-1].map(grid)
        gridSolution = galerkinMethodObject.evaluateSolutionAtPoints(mappedGrid)
        calculatedDimensionless_BPL_asol = dimensionless_BPL_asol(mappedGrid, gamma, beta)
        integrable = galerkinMethodObject.elements[-1].inverseDerivativeMap(mappedGrid) * \
                                    (gridSolution - calculatedDimensionless_BPL_asol)**2
        error += np.sum(w * integrable)
        errors = np.append(errors, np.sum(w * integrable))

    nonZeroAmount = galerkinMethodObject.getAmountOfNonZeroSLAE_elements()
    return nonZeroAmount, error, errors


indices = []
errors = []

amountOfAdditionalIntervalBefore1 = 5
amountOfAdditionalIntervalAfter1 = 5
MeshBefore1 = np.array([0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 0.01])
MeshAfter1 = np.array([1.0, np.inf])
curApproxOrders = 2*np.ones([MeshBefore1.size + MeshAfter1.size - 1], dtype=int)
# curApproxOrders = np.arrar([])
# curApproxOrders[-1] = 10

# bounds = (amountOfAdditionalInterval + 1) * [(0.0, 1.0)]
RangeBefore1 = np.arange(0, amountOfAdditionalIntervalBefore1)
RangeAfter1 = np.arange(1, amountOfAdditionalIntervalAfter1) + 1
# mesh = np.hstack([0.0, *np.cumsum(x[:amountOfAdditionalInterval]), np.inf])
# MeshBefore1 = np.hstack([0.0, (((1 / 4) ** (RangeBefore1))[::-1])])

# MeshAfter1 = np.hstack([RangeAfter1 * 1, np.inf])


Mesh = np.hstack([MeshBefore1, MeshAfter1])
print("mesh", Mesh)
elemTypes = np.zeros(MeshBefore1.size + MeshAfter1.size - 1, dtype=int)
elemTypes[-1] = 1

# totalApproxWeight = np.sum(x)
        # approxOrdersFloat = approxOrdersAmount*np.array([x/totalApproxWeight], dtype=float)
        # approxOrders = np.array(approxOrdersFloat, dtype=int)




for i in range(0, 100, 1):
    indices.append(i)
    nonZero, Max, errors = fun(meshArg=Mesh, approximationOrder=np.squeeze(curApproxOrders), mappingType=elemTypes,
                 gamma=2, beta=4, integrationPointsAmount=2000)
    print(nonZero, Max, *curApproxOrders, *errors)
    curApproxOrders += np.ones(curApproxOrders.size, dtype=int)
    # def lambdaFun(param: float, maxParam: int):
    #     # tmpApproxOrders = approxOrders.copy()
    #     approxOrders = 2*np.ones([2], dtype=int)
    #     approxOrders[0] += int(param)
    #     approxOrders[1] += maxParam - int(param)
    #     # print(approxOrders)
    #     result = fun(meshArg=Mesh, approximationOrder=np.squeeze(approxOrders), mappingType=elemTypes,
    #                  gamma=1, beta=4, integrationPointsAmount=2000)[1]
    #     return result

    # argMax = 0
    # nonZero, Max, errors = lambdaFun(curApproxOrders, 0)

    # searchResult = sp_opt.direct(lambda x: lambdaFun(x, i + 1), bounds=[(0, i + 1)], maxiter=10)
    # print(searchResult)
    # for perturbedOrderIter in range(1, MeshBefore1.size + MeshAfter1.size - 1):
    #     nonZero, error, errors = lambdaFun(curApproxOrders, perturbedOrderIter)
    #     if error < Max:
    #         Max = error
    #         argMax = perturbedOrderIter
    # curApproxOrders[argMax] += 1
    # print(errors)
    # plt.plot(errors)
    # plt.show()
    # print(nonZero, Max, *curApproxOrders, *errors)


plt.loglog(indices, errors)
plt.show()