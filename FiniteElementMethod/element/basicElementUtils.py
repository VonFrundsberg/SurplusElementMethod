import time

from FiniteElementMethod.element.basicElement import basicElement as belem
import numpy as np
import mathematics.approximate as approx
import mathematics.integrate as integr
def integrateBilinearForm1(trialElement: belem, testElement: belem,
                           weight, integrationPointsAmount: int):
    """Integrates bilinear form of the type a(u, v) = int_K weight(x) du(x) dv(x) dx,
        where u(x) and v(x) are basis functions of trial and test elements respectively
        integration is performed over K := {non-zero interval of trial element}

        Arguments:
            trialElement: basicElement type
            testElement: basicElement type
            weight: function: R->R
            integrationPointsAmount: regulates precision of numerical integration
        Returns:
            result: an integral of specified bilinear form
    """

    w, x = integr.reg_32_wn(a=-1, b=1, n=integrationPointsAmount)
    w = w*trialElement.inverseDerivativeMap(x)
    trialD = trialElement.evalDiff(x)
    testD = testElement.evalDiff(x)

    mappedX = trialElement.map(x)

    W = weight(mappedX)*w
    D2 = np.einsum('ij,ik->ijk', trialD, testD)


    resultIntegrals = np.einsum('ijk, i -> jk', D2, W)
    return resultIntegrals


def evaluateBilinearFormAtBoundary2(trialElement: belem, testElement: belem,
                                    weight):
    """Evaluates bilinear form of the type
        a(u, v) = weight(R) grad u(R) v(R) + weight(L) grad u(L) v(L),
        where u(x) and v(x) are basis functions of elementU element
        L and R are left and right (in inner limit) boundary of elementU interval.

        Arguments:
            elementU:
            weight:
        Returns:
            result: evaluated differences at boundaries
    """

    if (np.max(np.abs(trialElement.interval - testElement.interval)) <= np.finfo(float).eps*10):
        leftBoundaryPoint = trialElement.interval[0]
        rightBoundaryPoint = trialElement.interval[1]

        leftRealSpacePoint = trialElement.inverseMap(leftBoundaryPoint)
        rightRealSpacePoint = trialElement.inverseMap(rightBoundaryPoint)

        leftWeight = weight(leftRealSpacePoint) * trialElement.inverseDerivativeMap(leftBoundaryPoint)
        rightWeight = weight(rightRealSpacePoint) * trialElement.inverseDerivativeMap(rightBoundaryPoint)

        leftTrialD = trialElement.evalDiff(leftBoundaryPoint)
        leftTestI = testElement.eval(leftBoundaryPoint)

        rightTrialD = trialElement.evalDiff(rightBoundaryPoint)
        rightTestI = testElement.eval(rightBoundaryPoint)
        leftEvaluation = leftWeight * np.einsum("ij, ik -> jk", leftTrialD, leftTestI)
        rightEvaluation = rightWeight * np.einsum("ij, ik -> jk", rightTrialD, rightTestI)
        result = rightEvaluation + leftEvaluation
        return result

    if(trialElement.interval[1] == testElement.interval[0]):
        """"Case, where trial functions are on the LHS of test functions"""
        rightBoundaryPoint = trialElement.interval[1]

        rightRealSpacePoint = trialElement.inverseMap(rightBoundaryPoint)

        rightWeight = weight(rightRealSpacePoint) * trialElement.inverseDerivativeMap(rightBoundaryPoint)

        rightTrialD = trialElement.evalDiff(rightBoundaryPoint)
        rightTestI = testElement.eval(rightBoundaryPoint)
        rightEvaluation = rightWeight * np.einsum("ij, ik -> jk", rightTrialD, rightTestI)
        result = rightEvaluation
        return result
    if(trialElement.interval[0] == testElement.interval[1]):
        """Case, where trial functions are on the RHS of test functions"""
        leftBoundaryPoint = trialElement.interval[0]

        leftRealSpacePoint = trialElement.inverseMap(leftBoundaryPoint)

        leftWeight = weight(leftRealSpacePoint) * trialElement.inverseDerivativeMap(leftBoundaryPoint)

        leftTrialD = trialElement.evalDiff(leftBoundaryPoint)
        leftTestI = testElement.eval(leftBoundaryPoint)

        leftEvaluation = leftWeight * np.einsum("ij, ik -> jk", leftTrialD, leftTestI)
        result = leftEvaluation
        return result

def integrateFunctional(testElement: belem, function, weight,
        integrationPointsAmount: int):
    """(one-dimensional) Integrates functional form of the type l(v) = int_K function(x) v_j(x) dx,
            where v_j(x) are basis functions of testElement
            and K is a non-zero region of testElement.

            Arguments:
                testElement:
                function:
                weight:
                integrationPointsAmount:
            Returns:
                result: an integral of functional
        """
    w, x = integr.reg_32_wn(a=-1, b=1, n=integrationPointsAmount)
    w = w*testElement.inverseDerivativeMap(x)
    I = testElement.eval(x)
    x = testElement.map(x)
    W = function(x)*w

    resultIntegrals = np.einsum('ij, i -> j', I, W)

    return resultIntegrals