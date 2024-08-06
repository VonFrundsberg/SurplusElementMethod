from SurplusElement.GalerkinMethod.element.mainElementClass import element
import numpy as np
from SurplusElement import mathematics as integr


def integrateBilinearForm0_TensorWeight(elementU: element, evaluatedFuncsTensor,
        integrationPointsAmount: int, axis: int, lambdaWeightAlongAxis = None):
    """
    """
    w, x = integr.reg_32_wn(a=-1, b=1, n=integrationPointsAmount)
    w = w * elementU[axis].inverseDerivativeMap(x)
    mappedIntegrationNodes = elementU[axis].map(x)
    evaluatedBasisFunctions = elementU[axis].eval(x)

    if lambdaWeightAlongAxis is not None:
        integrWeight = lambdaWeightAlongAxis(mappedIntegrationNodes) * w
    else:
        integrWeight = w


    matrixFunction = np.einsum('ij,ik->ijk',
                               evaluatedBasisFunctions, evaluatedBasisFunctions)
    # print("basis funcs, potential shape")
    # print(matrixFunction.shape)
    # print(evaluatedFuncsTensor.shape)
    resultIntegrals = np.einsum('iln, iamb, i -> lnamb',
                                matrixFunction, evaluatedFuncsTensor, integrWeight)
    # print("result shape")
    # print(resultIntegrals.shape)
    return resultIntegrals
def integrateBilinearForm0(elementU: element, weight, integrationPointsAmount: int, axis: int):
    """(one-dimensional) Integrates bilinear form of the type a(u, u) = int_K weight(x) u(x) v(x) dx,
        where u(x) and v(x) are basis functions of elementU element
        and K is a non-zero region of elementU functions.
        weight(x) must be a calculable function on an element grid

        Arguments:
            elementU:
            weight:
            integrationPointsAmount:
        Returns:
            result: an integral of chosen bilinear form and elementU
    """
    w, x = integr.reg_32_wn(a=-1, b=1, n=integrationPointsAmount)
    w = w * elementU[axis].inverseDerivativeMap(x)
    D = elementU[axis].eval(x)
    x = elementU[axis].map(x)
    W = weight(x) * w
    D2 = np.einsum('ij,ik->ijk', D, D)

    resultIntegrals = np.einsum('ijk, i -> jk', D2, W)
    return resultIntegrals
def integrateBilinearForm1_SameElement(elementU: element, weight, integrationPointsAmount: int, axis: int):
    """(one-dimensional) Integrates bilinear form of the type a(u, v) = int_K weight(x) du(x) dv(x) dx,
        where U(x) and v(x) are basis functions of elementU element
        and K is a non-zero region of elementU functions.

        Arguments:
            elementU:
            weight:
            integrationPointsAmount:
            axis:
        Returns:
            result: an integral of chosen bilinear form and elementU
    """

    w, x = integr.reg_32_wn(a=-1, b=1, n=integrationPointsAmount)
    w = w*elementU[axis].inverseDerivativeMap(x)
    D = elementU[axis].evalDiff(x)
    x = elementU[axis].map(x)

    W = weight(x)*w
    D2 = np.einsum('ij,ik->ijk', D, D)


    resultIntegrals = np.einsum('ijk, i -> jk', D2, W)
    return resultIntegrals
def integrateBilinearForm2(elementU: element, weight, integrationPointsAmount: int, axis:int):
    """(one-dimensional) Integrates bilinear form of the type
        a(u, u) = int_K weight(x) grad u(x) v(x) dx,
        where U(x) and v(x) are basis functions of elementU element
        and K is a non-zero region of elementU functions.

        Arguments:
            elementU:
            weight:
            integrationPointsAmount:
        Returns:
            result: an integral of chosen bilinear form and elementU
    """
    w, x = integr.reg_32_wn(a=-1, b=1, n=integrationPointsAmount)
    w = w * elementU[axis].inverseDerivativeMap(x)
    D = elementU[axis].evalDiff(x)
    I = elementU[axis].eval(x)
    x = elementU[axis].map(x)

    W = weight(x) * w
    D2 = np.einsum('ij,ik->ijk', I, D)

    resultIntegrals = np.einsum('ijk, i -> jk', D2, W)
    return resultIntegrals

def integrateBilinearForm3(elementU: element, weight, integrationPointsAmount: int):
    """4-dimensional Integrates bilinear form of the type
        a(u, u) = int_K weight(x)(grad_1 u(x) grad_1 v(x) + grad_2 u(x) grad_2 v(x))dx,
        where U(x) and v(x) are basis functions of elementU element
        and K is a non-zero region of elementU functions.

        Arguments:
            elementU:
            weight:
            integrationPointsAmount:
        Returns:
            result: an integral of chosen bilinear form and elementU
    """
    return None



def integrateFunctional(elementU: element, function,
        integrationPointsAmount: int, axis: int, ttForm=False, precalc=False):
    """(one-dimensional) Integrates functional form of the type l(v) = int_K function(x) v(x) dx,
            where v(x) are basis functions of elementU element
            and K is a non-zero region of elementU functions.

            Arguments:
                elementU:
                weight:
                integrationPointsAmount:
                axis:
            Returns:
                result: an integral of functional
        """
    if precalc == False:
        if ttForm == False:
            w, x = integr.reg_32_wn(a=-1, b=1, n=integrationPointsAmount)
            w = w*elementU[axis].inverseDerivativeMap(x)
            I = elementU[axis].eval(x)
            x = elementU[axis].map(x)
            W = function(x)*w

            resultIntegrals = np.einsum('ij, i -> j', I, W)
        elif ttForm == True:
            w, x = integr.reg_32_wn(a=-1, b=1, n=integrationPointsAmount)
            w = w * elementU[axis].inverseDerivativeMap(x)
            W = np.einsum('ij, i -> ij', function(x), w)
            I = elementU[axis].eval(x)
            resultIntegrals = np.einsum('ij, ik -> kj', I, W)
    elif precalc == True:
        w, x = integr.reg_32_wn(a=-1, b=1, n=function.shape[0] - 32)
        w = w * elementU[axis].inverseDerivativeMap(x)
        # print(function.shape, w.shape)
        W = np.einsum('ij, i -> ij', function, w)
        I = elementU[axis].eval(x)
        resultIntegrals = np.einsum('ij, ik -> jk', I, W)
    return resultIntegrals

def integrateFunctionalWithMatrixRHS(elementU: element, evaluatedFuncsList,
        integrationPointsAmount: int, axis: int, lambdaWeightAlongAxis = None):
    """(one-dimensional) Integrates functional form of the type
        l(v) := int_K matrixFunction(i, j, x) v(x) dx,
        where matrix function i,j is based on functions from evaluatedFuncsList
        as all combinations of products of i and j functions from list

            Arguments:
                elementU:
                weight: optional weight along chosen axis
                integrationPointsAmount:
                evaluatedFuncsList: list of functions to be evaluated with shape=(integrationPointsAmount, n)
                axis:
            Returns:
                result: an integral of functional calculated by the following formula
                in, ijk, i -> njk
                where "in" is the shape of basis functions
                    "ijk" is the shape of matrixFunctions
                    "i" is the shape of calculated weight function
        """
    w, x = integr.reg_32_wn(a=-1, b=1, n=integrationPointsAmount)
    w = w * elementU[axis].inverseDerivativeMap(x)
    evaluatedBasisFunctions = elementU[axis].eval(x)
    mappedIntegrationNodes = elementU[axis].map(x)
    if lambdaWeightAlongAxis is not None:
        integrWeight = lambdaWeightAlongAxis(mappedIntegrationNodes) * w
    else:
        integrWeight = w


    matrixFunction = np.einsum('ij,ik->ijk', evaluatedFuncsList, evaluatedFuncsList)
    resultIntegrals = np.einsum('in, ijk, i -> njk',
                                evaluatedBasisFunctions, matrixFunction, integrWeight)
    return resultIntegrals