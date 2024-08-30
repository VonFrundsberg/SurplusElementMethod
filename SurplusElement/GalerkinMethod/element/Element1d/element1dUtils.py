from SurplusElement.GalerkinMethod.element.Element1d.element1d import Element1d as belem
import numpy as np
from SurplusElement import mathematics as integr


def integrateBilinearForm0(trialElement: belem, testElement: belem,
                           weight, integrationPointsAmount: int):
    """Integrates bilinear form of the type a(u, v) = int_K weight(x) u(x) v(x) dx,
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
    mappedX = trialElement.map(x)
    w = w * trialElement.inverseDerivativeMap(x)
    trialD = trialElement.eval(mappedX)
    testD = testElement.eval(mappedX)
    W = w * weight(mappedX)
    D2 = np.einsum('ij,ik->ijk', trialD, testD)


    resultIntegrals = np.einsum('ijk, i -> jk', D2, W)
    return resultIntegrals
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
    mappedX = trialElement.map(x)
    w = w * trialElement.inverseDerivativeMap(x)
    trialD = trialElement.evalDiff(mappedX)
    testD = testElement.evalDiff(mappedX)
    W = w * weight(mappedX)
    D2 = np.einsum('ij,ik->ijk', trialD, testD)

    resultIntegrals = np.einsum('ijk, i -> jk', D2, W)
    return resultIntegrals

def integrateBilinearForm2(trialElement: belem, testElement: belem,
                           weight, integrationPointsAmount: int):
    """Integrates bilinear form of the type a(u, v) = int_K weight(x) du(x) v(x) dx,
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
    mappedX = trialElement.map(x)
    w = w * trialElement.inverseDerivativeMap(x)
    trialD = trialElement.evalDiff(mappedX)
    testD = testElement.eval(mappedX)
    W = w * weight(mappedX)
    D2 = np.einsum('ij,ik->ijk', trialD, testD)

    resultIntegrals = np.einsum('ijk, i -> jk', D2, W)
    return resultIntegrals



def evaluateDG_JumpComponentMain(trialElement: belem, testElement: belem,
                                 weight):
    """Evaluates bilinear form of the type
        a(u, v) = weight(R) 0.5 * [du(R-) + du(R+)] [v(R-) - v(R+)) +
                    weight(L) 0.5 * [du(L-) + du(L+)] [v(L-) - v(L+)),
        where u(x)  are basis functions of trialElement
        and v(x)  are basis functions of testElement
        L and R are left and right (in inner limit) boundary of elementU interval.

        Arguments:
            trialElement:
            testElement:
            weight:
        Returns:
            result: evaluated values at boundaries
    """
    leftRef_LeftLim = np.nextafter(trialElement.interval[0], -np.inf)
    leftRef_RightLim = np.nextafter(trialElement.interval[0], np.inf)
    rightRef_LeftLim = np.nextafter(trialElement.interval[1], -np.inf)
    rightRef_RightLim = np.nextafter(trialElement.interval[1], np.inf)

    leftWeight = 0.5 * weight(trialElement.interval[0])  # * trialElement.inverseDerivativeMap(leftBoundaryPoint)
    rightWeight = 0.5 * weight(trialElement.interval[1])  # * trialElement.inverseDerivativeMap(rightBoundaryPoint)

    output = False
    if output:
        print("Main DG component")
        if trialElement.interval[1] == np.inf:
            print("Intervals for trial and test elements are: ")
            print(trialElement.interval, testElement.interval)

            print("Left and right limits of diff TRIAL LEFT point are:")
            print(trialElement.evalDiff(leftRef_LeftLim))
            # print(leftRef_RightLim)
            print(trialElement.evalDiff(leftRef_RightLim))

            print("Left and right limits of diff TEST LEFT point are:")
            print(testElement.evalDiff(leftRef_LeftLim))
            print(testElement.evalDiff(leftRef_RightLim))

            print("Left and right limits of diff TRIAL RIGHT point are:")
            print(trialElement.evalDiff(rightRef_LeftLim))
            print(trialElement.evalDiff(rightRef_RightLim))

            print("Left and right limits of diff TEST RIGHT point are:")
            print(testElement.evalDiff(rightRef_LeftLim))
            print(testElement.evalDiff(rightRef_RightLim))

            print("Weight at left and right points")
            print(leftWeight, rightWeight)





    leftTrialD = trialElement.evalDiff(leftRef_LeftLim) + trialElement.evalDiff(leftRef_RightLim)
    leftTestI = testElement.eval(leftRef_LeftLim) - testElement.eval(leftRef_RightLim)

    rightTrialD = trialElement.evalDiff(rightRef_LeftLim) + trialElement.evalDiff(rightRef_RightLim)
    rightTestI = testElement.eval(rightRef_LeftLim) - testElement.eval(rightRef_RightLim)

    leftEvaluation = leftWeight * np.einsum("ij, ik -> jk", leftTrialD, leftTestI)
    rightEvaluation = rightWeight * np.einsum("ij, ik -> jk", rightTrialD, rightTestI)

    nansLeft = np.isnan(leftEvaluation)
    leftEvaluation[nansLeft] = 0.0

    nansRight = np.isnan(rightEvaluation)
    rightEvaluation[nansRight] = 0.0
    result = rightEvaluation + leftEvaluation
    return result
def evaluateDG_JumpComponentSymmetry(trialElement: belem, testElement: belem,
                                 weight):
    """Evaluates bilinear form of the type
        a(u, v) = weight(R) 0.5 * [u(R-) - u(R+)] [dv(R-) + dv(R+)) +
                    weight(L) 0.5 * [u(L-) - u(L+)] [dv(L-) + dv(L+)),
        where u(x)  are basis functions of trialElement
        and v(x)  are basis functions of testElement
        L and R are left and right (in inner limit) boundary of elementU interval.

        Arguments:
            trialElement:
            testElement:
            weight:
        Returns:
            result: evaluated differences at boundaries
    """
    leftRef_LeftLim = np.nextafter(trialElement.interval[0], -np.inf)
    leftRef_RightLim = np.nextafter(trialElement.interval[0], np.inf)
    rightRef_LeftLim = np.nextafter(trialElement.interval[1], -np.inf)
    rightRef_RightLim = np.nextafter(trialElement.interval[1], np.inf)

    leftWeight = 0.5 * weight(trialElement.interval[0])  # * trialElement.inverseDerivativeMap(leftBoundaryPoint)
    rightWeight = 0.5 * weight(trialElement.interval[1])  # * trialElement.inverseDerivativeMap(rightBoundaryPoint)

    leftTrialI = trialElement.eval(leftRef_LeftLim) - trialElement.eval(leftRef_RightLim)
    leftTestD = testElement.evalDiff(leftRef_LeftLim) + testElement.evalDiff(leftRef_RightLim)

    rightTrialI = trialElement.eval(rightRef_LeftLim) - trialElement.eval(rightRef_RightLim)
    rightTestD = testElement.evalDiff(rightRef_LeftLim) + testElement.evalDiff(rightRef_RightLim)

    leftEvaluation = leftWeight * np.einsum("ij, ik -> jk", leftTrialI, leftTestD)
    rightEvaluation = rightWeight * np.einsum("ij, ik -> jk", rightTrialI, rightTestD)

    nansLeft = np.isnan(leftEvaluation)
    leftEvaluation[nansLeft] = 0.0

    nansRight = np.isnan(rightEvaluation)
    rightEvaluation[nansRight] = 0.0

    result = rightEvaluation + leftEvaluation
    return result


def evaluateDG_ErrorComponent(trialElement: belem, testElement: belem,
                                 weight):
    """Evaluates bilinear form of the type
        a(u, v) = weight(R) [u(R-) - u(R+)] * [v(R-) - v(R+)] +
         weight(L) [u(L-) - u(L+)] [v(L-) - v(L+)),
        where u(x)  are basis functions of trialElement
        and v(x)  are basis functions of testElement
        L and R are left and right (in inner limit) boundary of elementU interval.

        Arguments:
            trialElement:
            testElement:
            weight:
        Returns:
            result: evaluated values at boundaries
    """
    leftRef_LeftLim = np.nextafter(trialElement.interval[0], -np.inf)
    leftRef_RightLim = np.nextafter(trialElement.interval[0], np.inf)
    rightRef_LeftLim = np.nextafter(trialElement.interval[1], -np.inf)
    rightRef_RightLim = np.nextafter(trialElement.interval[1], np.inf)

    leftWeight = weight(trialElement.interval[0])  # * trialElement.inverseDerivativeMap(leftBoundaryPoint)
    rightWeight = weight(trialElement.interval[1])  # * trialElement.inverseDerivativeMap(rightBoundaryPoint)

    output = False
    if output:
        print("Main DG component")
        if trialElement.interval[1] == np.inf:
            print("Intervals for trial and test elements are: ")
            print(trialElement.interval, testElement.interval)

            print("Left and right limits of diff TRIAL LEFT point are:")
            print(trialElement.evalDiff(leftRef_LeftLim))
            # print(leftRef_RightLim)
            print(trialElement.evalDiff(leftRef_RightLim))

            print("Left and right limits of diff TEST LEFT point are:")
            print(testElement.evalDiff(leftRef_LeftLim))
            print(testElement.evalDiff(leftRef_RightLim))

            print("Left and right limits of diff TRIAL RIGHT point are:")
            print(trialElement.evalDiff(rightRef_LeftLim))
            print(trialElement.evalDiff(rightRef_RightLim))

            print("Left and right limits of diff TEST RIGHT point are:")
            print(testElement.evalDiff(rightRef_LeftLim))
            print(testElement.evalDiff(rightRef_RightLim))

            print("Weight at left and right points")
            print(leftWeight, rightWeight)

    leftTrialI = trialElement.eval(leftRef_LeftLim) - trialElement.eval(leftRef_RightLim)
    leftTestI = testElement.eval(leftRef_LeftLim) - testElement.eval(leftRef_RightLim)

    rightTrialI = trialElement.eval(rightRef_LeftLim) - trialElement.eval(rightRef_RightLim)
    rightTestI = testElement.eval(rightRef_LeftLim) - testElement.eval(rightRef_RightLim)

    leftEvaluation = leftWeight * np.einsum("ij, ik -> jk",
                                            leftTrialI, leftTestI)
    rightEvaluation = rightWeight * np.einsum("ij, ik -> jk",
                                              rightTrialI, rightTestI)

    nansLeft = np.isnan(leftEvaluation)
    leftEvaluation[nansLeft] = 0.0

    nansRight = np.isnan(rightEvaluation)
    rightEvaluation[nansRight] = 0.0
    result = rightEvaluation + leftEvaluation
    return result
def evaluateBilinearFormAtBoundary_20(trialElement: belem, testElement: belem, weight):
    """Evaluates bilinear form of the type
        a(u, v) = weight(R) grad u(R) v(R) + weight(L) grad u(L) v(L),
        where u(x) and v(x) are basis functions of elementU element
        L and R are left and right (in inner limit) boundary of elementU interval.

        Arguments:
            trialElement:
            testElement:
            weight:
        Returns:
            result: evaluated differences at boundaries
    """

    if (np.max(np.abs(trialElement.interval - testElement.interval)) <= np.finfo(float).eps*10):
        epsilon = np.finfo(float).eps
        leftBoundaryPoint = -1
        rightBoundaryPoint = 1
        # leftRealSpacePoint = trialElement.inverseMap(leftBoundaryPoint)
        # rightRealSpacePoint = trialElement.inverseMap(rightBoundaryPoint)

        leftWeight = weight(trialElement.interval[0]) #* trialElement.inverseDerivativeMap(leftBoundaryPoint)
        rightWeight = weight(trialElement.interval[1]) #* trialElement.inverseDerivativeMap(rightBoundaryPoint)

        leftTrialD = trialElement.evalDiff(leftBoundaryPoint)
        leftTestI = testElement.eval(leftBoundaryPoint)

        rightTrialD = trialElement.evalDiff(rightBoundaryPoint)
        rightTestI = testElement.eval(rightBoundaryPoint)
        leftEvaluation = leftWeight * np.einsum("ij, ik -> jk", leftTrialD, leftTestI)
        rightEvaluation = rightWeight * np.einsum("ij, ik -> jk", rightTrialD, rightTestI)
        result = rightEvaluation + leftEvaluation
        return result

    # if(trialElement.interval[1] == testElement.interval[0]):
    #     """"Case, where trial functions are on the LHS of test functions"""
    #     rightBoundaryPoint = trialElement.interval[1]
    #
    #     rightRealSpacePoint = trialElement.inverseMap(rightBoundaryPoint)
    #
    #     rightWeight = weight(rightRealSpacePoint) * trialElement.inverseDerivativeMap(rightBoundaryPoint)
    #
    #     rightTrialD = trialElement.evalDiff(rightBoundaryPoint)
    #     rightTestI = testElement.eval(rightBoundaryPoint)
    #     rightEvaluation = rightWeight * np.einsum("ij, ik -> jk", rightTrialD, rightTestI)
    #     result = rightEvaluation
    #     return result
    # if(trialElement.interval[0] == testElement.interval[1]):
    #     """Case, where trial functions are on the RHS of test functions"""
    #     leftBoundaryPoint = trialElement.interval[0]
    #
    #     leftRealSpacePoint = trialElement.inverseMap(leftBoundaryPoint)
    #
    #     leftWeight = weight(leftRealSpacePoint) * trialElement.inverseDerivativeMap(leftBoundaryPoint)
    #
    #     leftTrialD = trialElement.evalDiff(leftBoundaryPoint)
    #     leftTestI = testElement.eval(leftBoundaryPoint)
    #
    #     leftEvaluation = leftWeight * np.einsum("ij, ik -> jk", leftTrialD, leftTestI)
    #     result = leftEvaluation
    #     return result
def evaluateBilinearFormAtBoundary_21(trialElement: belem, testElement: belem, weight):
    """Evaluates bilinear form of the type
        a(u, v) = weight(R) u(R) grad v(R) + weight(L) u(L) grad v(L),
        where u(x) and v(x) are basis functions of elementU element
        L and R are left and right (in inner limit) boundary of elementU interval.

        Arguments:
            trialElement:
            testElement:
            weight:
        Returns:
            result: evaluated differences at boundaries
    """

    if (np.max(np.abs(trialElement.interval - testElement.interval)) <= np.finfo(float).eps*10):
        leftBoundaryPoint = trialElement.interval[0]
        rightBoundaryPoint = trialElement.interval[1]

        leftRealSpacePoint = trialElement.inverseMap(leftBoundaryPoint)
        rightRealSpacePoint = trialElement.inverseMap(rightBoundaryPoint)

        leftWeight = weight(leftRealSpacePoint)# * trialElement.inverseDerivativeMap(leftBoundaryPoint)
        rightWeight = weight(rightRealSpacePoint)# * trialElement.inverseDerivativeMap(rightBoundaryPoint)

        leftTrialI = trialElement.eval(leftBoundaryPoint)
        leftTestD = testElement.evalDiff(leftBoundaryPoint)

        rightTrialI = trialElement.eval(rightBoundaryPoint)
        rightTestD = testElement.evalDiff(rightBoundaryPoint)
        leftEvaluation = leftWeight * np.einsum("ij, ik -> jk", leftTrialI, leftTestD)
        rightEvaluation = rightWeight * np.einsum("ij, ik -> jk", rightTrialI, rightTestD)
        result = rightEvaluation + leftEvaluation
        return result

    if(trialElement.interval[1] == testElement.interval[0]):
        """"Case, where trial functions are on the LHS of test functions"""
        rightBoundaryPoint = trialElement.interval[1]

        rightRealSpacePoint = trialElement.inverseMap(rightBoundaryPoint)

        rightWeight = weight(rightRealSpacePoint) * trialElement.inverseDerivativeMap(rightBoundaryPoint)

        rightTrialI = trialElement.eval(rightBoundaryPoint)
        rightTestD = testElement.evalDiff(rightBoundaryPoint)
        rightEvaluation = rightWeight * np.einsum("ij, ik -> jk", rightTrialI, rightTestD)
        result = rightEvaluation
        return result

    if(trialElement.interval[0] == testElement.interval[1]):
        """Case, where trial functions are on the RHS of test functions"""
        leftBoundaryPoint = trialElement.interval[0]

        leftRealSpacePoint = trialElement.inverseMap(leftBoundaryPoint)

        leftWeight = weight(leftRealSpacePoint) * trialElement.inverseDerivativeMap(leftBoundaryPoint)

        leftTrialI = trialElement.evalDiff(leftBoundaryPoint)
        leftTestD = testElement.eval(leftBoundaryPoint)

        leftEvaluation = leftWeight * np.einsum("ij, ik -> jk", leftTrialI, leftTestD)
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
    w, x = integr.reg_22_wn(a=-1, b=1, n=integrationPointsAmount)
    w = w*testElement.inverseDerivativeMap(x)

    x = testElement.map(x)
    I = testElement.eval(x)
    W = np.nan_to_num(function(x)*w*weight(x))
    resultIntegrals = np.einsum('ij, i -> j', I, W)

    return resultIntegrals

def integrateTensorFunctional(testElement: belem, function, tensorShape, weight,
        integrationPointsAmount: int):
    """(one-dimensional) Integrates functional form of the type l(v) = int_K function(x) v_j(x) dx,
            where v_j(x) are basis functions of testElement,
            K is a non-zero region of testElement,
            function is R -> R^{tensorShape}

            Arguments:
                testElement:
                function:
                weight:
                integrationPointsAmount:
            Returns:
                result: an integral of functional
        """
    w, x = integr.reg_22_wn(a=-1, b=1, n=integrationPointsAmount)
    w = w*testElement.inverseDerivativeMap(x)

    x = testElement.map(x)
    I = testElement.eval(x)
    W = np.nan_to_num(function(x)*w*weight(x))
    W = np.reshape(W, [np.prod(tensorShape), W.shape[-1]])
    resultIntegrals = np.einsum('ij, ki -> jk', I, W)
    return np.reshape(resultIntegrals, [I.shape[1], *tensorShape])