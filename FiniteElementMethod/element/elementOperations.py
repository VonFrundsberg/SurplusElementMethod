import time

from FiniteElementMethod.element.mainElementClass import element
import numpy as np
import mathematics.approximate as approx
import mathematics.integrate as integr
def integrateBilinearForm0(elementU: element,weight, integrationPointsAmount: int, axis: int):
    """(one-dimensional) Integrates bilinear form of the type a(u, u) = int_K weight(x) u(x) v(x) dx,
        where U(x) and v(x) are basis functions of elementU element
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
def integrateBilinearForm1(elementU: element, weight, integrationPointsAmount: int, axis: int):
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
def integrateBilinearForm2(elementU: element, weight, integrationPointsAmount: int):
    """(two-dimensional) Integrates bilinear form of the type
        a(u, u) = int_K weight(x) grad u(x) grad v(x) dx,
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

def integrateFunctional(elementU: element, function, integrationPointsAmount: int, axis: int):
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
    w, x = integr.reg_32_wn(a=-1, b=1, n=integrationPointsAmount)
    w = w*elementU[axis].inverseDerivativeMap(x)
    I = elementU[axis].eval(x)
    x = elementU[axis].map(x)
    W = function(x)*w

    resultIntegrals = np.einsum('ij, i -> j', I, W)
    return resultIntegrals

