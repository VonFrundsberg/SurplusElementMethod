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
    # epsilon = np.finfo(float).eps
    print(trialElement.interval)
    print(testElement.interval)
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
    # if(elementU[axis].interval == elementV[axis].interval):
    #     """"Case, where trial and test functions intervals overlap"""
    #     leftBoundaryPoint = elementU[axis].interval[0]
    #     rightBoundaryPoint = elementU[axis].interval[1]
    #
    #     leftRealSpacePoint = elementU[axis].inverseMap(leftBoundaryPoint)
    #     rightRealSpacePoint = elementU[axis].inverseMap(rightBoundaryPoint)
    #
    #     leftW = weight(leftRealSpacePoint) * elementU[axis].inverseDerivativeMap(leftBoundaryPoint)
    #     rightW = weight(rightRealSpacePoint) * elementU[axis].inverseDerivativeMap(rightBoundaryPoint)
    #
    #     leftU_D = elementU[axis].evalDiff(leftBoundaryPoint)
    #     leftV_I = elementV[axis].eval(leftBoundaryPoint)
    #
    #     rightU_D = elementU[axis].evalDiff(rightBoundaryPoint)
    #     rightV_I = elementV[axis].eval(rightBoundaryPoint)
    #
    #     leftEvaluation = np.einsum("i, j, k -> jk", leftW, leftU_D, leftV_I)
    #     rightEvaluation = np.einsum("i, j, k -> jk", rightW, rightU_D, rightV_I)
    #     return leftEvaluation - rightEvaluation
    #
    # if(elementU[axis].interval[1] == elementV[axis].interval[0]):
    #     """"Case, where trial functions are on the RHS of test functions"""
    #     rightBoundaryPoint = elementU[axis].interval[1]
    #
    #     rightRealSpacePoint = elementU[axis].inverseMap(rightBoundaryPoint)
    #
    #     rightW = weight(rightRealSpacePoint) * elementU[axis].inverseDerivativeMap(rightBoundaryPoint)
    #
    #     rightU_D = elementU[axis].evalDiff(rightBoundaryPoint)
    #     rightV_I = elementV[axis].eval(rightBoundaryPoint)
    #
    #     rightEvaluation = np.einsum("i, j, k -> jk", rightW, rightU_D, rightV_I)
    #     return rightEvaluation
    #
    # if (elementU[axis].interval[0] == elementV[axis].interval[1]):
    #     """"Case, where trial functions are on the LHS of test functions"""
    #     leftBoundaryPoint = elementU[axis].interval[0]
    #
    #     leftRealSpacePoint = elementU[axis].inverseMap(leftBoundaryPoint)
    #
    #     leftW = weight(leftRealSpacePoint) * elementU[axis].inverseDerivativeMap(leftBoundaryPoint)
    #     leftU_D = elementU[axis].evalDiff(leftBoundaryPoint)
    #     leftV_I = elementV[axis].eval(leftBoundaryPoint)
    #
    #
    #     leftEvaluation = np.einsum("i, j, k -> jk", leftW, leftU_D, leftV_I)
    #     return -leftEvaluation
