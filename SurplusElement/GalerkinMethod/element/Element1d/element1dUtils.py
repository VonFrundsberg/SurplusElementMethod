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


def integrateBilinearForm0weightTensor(trialElement: belem, testElement: belem,
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
    resultIntegrals = np.einsum('ijk, il -> jkl', D2, W)
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
    """Integrates bilinear form of the type a(u, v) = int_K weight(x) u(x) dv(x) dx,
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
    trialI = trialElement.eval(mappedX)
    testD = testElement.evalDiff(mappedX)
    W = w * weight(mappedX)
    D2 = np.einsum('ij,ik->ijk', trialI, testD)

    resultIntegrals = np.einsum('ijk, i -> jk', D2, W)
    return resultIntegrals



def evaluateDG_JumpComponentMain(trialElement: belem, testElement: belem,
                                 weight, physicalBoundary: np.ndarray, eps: float = 1e-16):
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
    leftRef_LeftLim = trialElement.interval[0] - eps
    leftRef_RightLim = np.nextafter(trialElement.interval[0], np.inf)
    leftRef_RightLim = trialElement.interval[0] + eps
    rightRef_LeftLim = np.nextafter(trialElement.interval[1], -np.inf)
    rightRef_LeftLim = trialElement.interval[1] - eps
    rightRef_RightLim = np.nextafter(trialElement.interval[1], np.inf)
    rightRef_RightLim = trialElement.interval[1] + eps
    # print(rightRef_LeftLim)
    leftWeight = 0.5 * weight(trialElement.interval[0])  # * trialElement.inverseDerivativeMap(leftBoundaryPoint)
    rightWeight = 0.5 * weight(trialElement.interval[1])  # * trialElement.inverseDerivativeMap(rightBoundaryPoint)
    if trialElement.interval[0] in physicalBoundary:
        leftWeight *= 0
    if trialElement.interval[1] in physicalBoundary:
        rightWeight *= 0
    output = False
    if output:
        print("Main DG component")
        # if trialElement.interval[1] == np.inf:
        if True:
            print("the p")
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
                                 weight, physicalBoundary: np.ndarray, eps: float = 1e-16):
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
    leftRef_LeftLim = trialElement.interval[0] - eps
    leftRef_RightLim = np.nextafter(trialElement.interval[0], np.inf)
    leftRef_RightLim = trialElement.interval[0] + eps
    rightRef_LeftLim = np.nextafter(trialElement.interval[1], -np.inf)
    rightRef_LeftLim = trialElement.interval[1] - eps
    rightRef_RightLim = np.nextafter(trialElement.interval[1], np.inf)
    rightRef_RightLim = trialElement.interval[1] + eps


    leftWeight = 0.5 * weight(trialElement.interval[0])  # * trialElement.inverseDerivativeMap(leftBoundaryPoint)
    rightWeight = 0.5 * weight(trialElement.interval[1])  # * trialElement.inverseDerivativeMap(rightBoundaryPoint)
    if trialElement.interval[0] in physicalBoundary:
        leftWeight *= 0
    if trialElement.interval[1] in physicalBoundary:
        rightWeight *= 0
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
                                 weight, physicalBoundary: np.ndarray, eps: float = 1e-16):
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
    leftRef_LeftLim = trialElement.interval[0] - eps
    leftRef_RightLim = np.nextafter(trialElement.interval[0], np.inf)
    leftRef_RightLim = trialElement.interval[0] + eps
    rightRef_LeftLim = np.nextafter(trialElement.interval[1], -np.inf)
    rightRef_LeftLim = trialElement.interval[1] - eps
    rightRef_RightLim = np.nextafter(trialElement.interval[1], np.inf)
    rightRef_RightLim = trialElement.interval[1] + eps

    leftWeight = weight(trialElement.interval[0])  # * trialElement.inverseDerivativeMap(leftBoundaryPoint)
    rightWeight = weight(trialElement.interval[1])  # * trialElement.inverseDerivativeMap(rightBoundaryPoint)
    if trialElement.interval[0] in physicalBoundary:
        leftWeight *= 0
    if trialElement.interval[1] in physicalBoundary:
        rightWeight *= 0
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


def evaluateDG_upwind(trialElement: belem, testElement: belem,
                                 weight, physicalBoundary: np.ndarray, eps: float = 1e-16):
    """Evaluates bilinear form of the type
        a(u, v) = weight(R) u(R-) * v(R+) -
         weight(L) u(L-) * v(L-),
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
    leftRef_LeftLim = trialElement.interval[0] - eps
    rightRef_RightLim = trialElement.interval[1] + eps

    leftWeight = weight(trialElement.interval[0])  # * trialElement.inverseDerivativeMap(leftBoundaryPoint)
    rightWeight = weight(trialElement.interval[1])  # * trialElement.inverseDerivativeMap(rightBoundaryPoint)

    leftTrialI = trialElement.eval(leftRef_LeftLim)
    leftTestI = testElement.eval(leftRef_LeftLim)

    rightTestI = testElement.eval(rightRef_RightLim)
    if trialElement.interval[0] in physicalBoundary:
        leftWeight *= 0
    if trialElement.interval[1] in physicalBoundary:
        rightWeight *= 0
    leftEvaluation = leftWeight * np.einsum("ij, ik -> jk",
                                            leftTrialI, leftTestI)
    rightEvaluation = rightWeight * np.einsum("ij, ik -> jk",
                                              leftTrialI, rightTestI)

    nansLeft = np.isnan(leftEvaluation)
    leftEvaluation[nansLeft] = 0.0

    nansRight = np.isnan(rightEvaluation)
    rightEvaluation[nansRight] = 0.0
    result = rightEvaluation + leftEvaluation
    return result


def evaluateBilinearFormAtBoundary0(trialElement: belem, testElement: belem, weight, B: float):
    """Evaluates bilinear form of the type
            a(u, v) = weight(B) du/dx(B) v(B),
            where u(x) and v(x) are basis functions of elementU element
            B is a point where boundary condition is enforced.

            Arguments:
                trialElement: belem
                testElement: belem
                weight: function
                B: float
            Returns:
                result: matrix with evaluated flux at the specified boundary
        """

    duX = trialElement.evalDiff(B)
    vX = testElement.eval(B)
    weightX = np.nan_to_num(weight(B), nan=0)
    return weightX * np.einsum("ij, ik -> jk", duX, vX)


def evaluateBilinearFormAtBoundary1(trialElement: belem, testElement: belem, weight, B: float):
    """Evaluates bilinear form of the type
            a(u, v) = weight(B) u(B) v(B),
            where u(x) and v(x) are basis functions of elementU element
            B is a point where boundary condition is enforced.

            Arguments:
                trialElement: belem
                testElement: belem
                weight: function
                B: float
            Returns:
                result: matrix with evaluated flux at the specified boundary
        """

    uX = trialElement.eval(B)
    vX = testElement.eval(B)
    weightX = np.nan_to_num(weight(B), nan=0)
    return weightX * np.einsum("ij, ik -> jk", uX, vX)

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
    # print('in')
    # if trialElement.interval.all() == testElement.interval.all():
    if (np.max(np.abs(trialElement.interval - testElement.interval)) <= np.finfo(float).eps * 10):
        # print('out')
        epsilon = np.finfo(float).eps
        leftBoundaryPoint = trialElement.interval[0]
        rightBoundaryPoint = trialElement.interval[1]
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
def evaluateBilinearFormAtBoundary_21(trialElement: belem, testElement: belem, weight):
    """Evaluates bilinear form of the type
        a(u, v) = weight(R) u(R) grad v(R) + weight(L) u(L) grad v(L),
        where u(x) and v(x) are basis functions of trialElement
        L and R are left and right (in inner limit) boundary of trialElement interval.

        Arguments:
            trialElement:
            testElement:
            weight:
        Returns:
            result: evaluated differences at boundaries
    """
    # print('in')
    # if trialElement.interval.all() == testElement.interval.all():
    if (np.max(np.abs(trialElement.interval - testElement.interval)) <= np.finfo(float).eps * 10):
        # print('out')
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

def evaluateFunctionalAtBoundaries1(testElement: belem,
                       leftBoundary, rightBoundary,  weight):
    """(one-dimensional) Evaluates functional of the type
    l(v) = weight(R) * v(R) + weight(L) * v(L),
            g_R = rightValue, g_L = leftValue
            L = leftBoundary, R = rightBoundary
        """

    vL = testElement.eval(leftBoundary) * weight(leftBoundary)
    vR = testElement.eval(rightBoundary) * weight(rightBoundary)
    return vR + vL

def evaluateFunctionalAtBoundaries0(testElement: belem,
                       leftBoundary, leftValue,
                       rightBoundary, rightValue, weight):
    """(one-dimensional) Evaluates functional of the type
    l(v) = weight(R) * g_R * v'(R) + weight(L) * g_L * v'(L),
            g_R = rightValue, g_L = leftValue
            L = leftBoundary, R = rightBoundary
        """

    vL = testElement.evalDiff(leftBoundary) * leftValue * weight(leftBoundary)
    vR = testElement.evalDiff(rightBoundary) * rightValue * weight(rightBoundary)
    return vR + vL

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