from FiniteElementMethod.element.mainElementClass import element
import FiniteElementMethod.element.elementOperations as elemOp
from mathematics import approximate as approx
import numpy as np
import scipy.linalg as sp_lin


def permutations(n):
    """Construct array of all permutations of arange(n) array"""
    a = np.zeros((np.math.factorial(n), n), np.uint8)
    f = 1
    for m in range(2, n + 1):
        b = a[:f, n - m + 1:]  # the block of permutations of range(m-1)
        for i in range(1, m):
            a[i * f:(i + 1) * f, n - m] = i
            a[i * f:(i + 1) * f, n - m + 1:] = b + (b >= i)
        b += 1
        f *= m
    return a
def testKronSum(elementU):
    """Some tests with kronecker sum of a differential-like operator, conversion to TT train format"""
    basisFuncList = []
    basisDiffFuncList = []
    for i in range(elementU.getDim()):
        basisFuncList.append(elementU[i].evalAtChebPoints()[1:-1, 1: -1])
        basisDiffFuncList.append(elementU[i].evalDiffAtChebPoints()[1:-1, 1: -1])
    C = np.kron(np.kron(basisDiffFuncList[0], basisFuncList[1]), basisFuncList[2])
    C += np.kron(np.kron(basisFuncList[0], basisDiffFuncList[1]), basisFuncList[2])
    C += np.kron(np.kron(basisFuncList[0], basisFuncList[1]), basisDiffFuncList[2])
    C += np.kron(np.kron(basisFuncList[0], basisFuncList[1]), basisFuncList[2])

    print(C.shape)
    old_C = C.copy()
    C = np.reshape(C, np.hstack((elementU.approxOrder - 2, elementU.approxOrder - 2)))
    C = np.transpose(C, axes=[0, 3, 1, 4, 2, 5])
    # C = np.transpose(C, axes=[0, 4, 1, 5, 2, 6, 3, 7])
    C = np.reshape(C, (elementU.approxOrder - 2)**2)
    C_TT_kron = approx.kronSumtoTT(basisFuncList, basisDiffFuncList)
    # for i in range(3):
    #     print(C_TT_1[i].shape)
    #     shape = C_TT_1[i].shape
    #     C_TT_1[i] = np.reshape(C_TT_1[i], [shape[0], shape[1]*shape[2], shape[3]])
    # if i == 0:
    #     print(i, "s core, upper")
    #     print(np.reshape(C_TT_2[i][0, :, 0], [3, 3]))
    #     print(i, "s core lower")
    #     print(np.reshape(C_TT_2[i][0, :, 1], [3, 3]))
    # shape = C_TT_2[i].shape
    # C_TT_2[i] = np.reshape(C_TT_2[i], [shape[0], shape[1] * shape[2], shape[3]])
    # print(C_TT_1[i].shape)
    # C = np.reshape(C, (elementU.approxOrder - 2)**2)
    # A = np.reshape(C_TT_1[0], [C_TT_1[0].shape[1], C_TT_1[0].shape[-1]])
    # initShape = C.shape
    # for i in range(1, len(C_TT_1)):
    #     shape = C_TT_1[i].shape
    #     reshaped = np.reshape(C_TT_1[i], [shape[0], shape[1] * shape[2]])
    #     A = np.dot(A, reshaped)
    #     A = np.reshape(A, [np.prod(initShape[:i + 1]), int(A.shape[-1]/initShape[i])])
    # A = np.reshape(A, initShape)
    # print("first", np.sum((A - C)**2))

    # A = np.reshape(C_TT_1[0], [C_TT_1[0].shape[1], C_TT_1[0].shape[-1]])
    # initShape = C.shape
    # for i in range(1, len(C_TT_1)):
    #     shape = C_TT_1[i].shape
    #     reshaped = np.reshape(C_TT_1[i], [shape[0], shape[1] * shape[2]])
    #     A = np.dot(A, reshaped)
    #     A = np.reshape(A, [np.prod(initShape[:i + 1]), int(A.shape[-1] / initShape[i])])
    # A = np.reshape(A, initShape)
    # print("second", np.sum((A - C) ** 2))
    # C = np.reshape(C, old_C.shape)
    # print("just for test", np.max(C - old_C))
    #
    # approx.kronSumtoTT()
def testTT_approximation(elementU, weight, TT_Tolerance):
    grid = elementU.getGrid()
    weightArr = weight(grid)
    weightArrTT = approx.simpleTTsvd(weightArr, tol=TT_Tolerance)
    print("elements amount before TT ", np.prod(weightArr.shape))
    sum = 0
    for it in weightArrTT:
        # print(it.shape, np.prod(it.shape))
        sum += np.prod(it.shape)
    A = np.reshape(weightArrTT[0], [weightArrTT[0].shape[1], weightArrTT[0].shape[-1]])
    initShape = weightArr.shape
    for i in range(1, len(weightArrTT)):
        shape = weightArrTT[i].shape
        reshaped = np.reshape(weightArrTT[i], [shape[0], shape[1] * shape[2]])
        A = np.dot(A, reshaped)
        A = np.reshape(A, [np.prod(initShape[:i + 1]), int(A.shape[-1]/initShape[i])])
    A = np.reshape(A, initShape)
    print(np.sum((A - weightArr)**2))
    print("after TT ", sum)

    binWeightArrTT = approx.simpleQTTsvd(weightArr, tol=1e-6)
    sum = 0
    for it in binWeightArrTT:
        # print(it.shape, np.prod(it.shape))
        sum += np.prod(it.shape)

    A = np.reshape(binWeightArrTT[0], [2, binWeightArrTT[0].shape[-1]])

    initShape = weightArr.shape
    for i in range(1, len(binWeightArrTT)):
        shape = binWeightArrTT[i].shape
        reshaped = np.reshape(binWeightArrTT[i], [shape[0], shape[1] * shape[2]])
        A = np.dot(A, reshaped)
        A = np.reshape(A, [2 ** (i + 1), int(A.shape[-1] / 2)])
    A = np.reshape(A, initShape)
    print(np.sum(np.abs(A - weightArr)))
    print("after binaryTT ", sum)
def testLaplacianInverseTT(elementU):
    basisFuncList = []
    basisDiffFuncList = []
    for i in range(elementU.getDim()):
        basisFuncList.append(elementU[i].evalAtChebPoints()[1:-1, 1: -1])
        basisDiffFuncList.append(elementU[i].evalDiffAtChebPoints()[1:-1, 1: -1])
    C = np.kron(np.kron(np.dot(basisDiffFuncList[0], basisDiffFuncList[0]), basisFuncList[1]), basisFuncList[2])
    C += np.kron(np.kron(basisFuncList[0], np.dot(basisDiffFuncList[1], basisDiffFuncList[1])), basisFuncList[2])
    C += np.kron(np.kron(basisFuncList[0], basisFuncList[1]), np.dot(basisDiffFuncList[2], basisDiffFuncList[2]))
    invC = sp_lin.inv(C)
    shape = elementU.approxOrder
    ttC = approx.matrixTTsvd(C, shape - 2, tol=1e-6)
    ttInvC = approx.matrixTTsvd(invC, shape - 2, tol=1e-3)
    print("nonzero amount in kron matrix: ", np.count_nonzero(C))
    print("nonzero amount in inverse matrix: ", np.count_nonzero(invC))
    sum = 0
    for i in range(len(ttC)):
        print("C core shape:", ttC[i].shape, "inverse C core shape ", ttInvC[i].shape, "with the total size"
                                                                                       "of ", ttInvC[i].size)
        sum += ttInvC[i].size
    print("inverse and TT approximation nonzero elements magnitude difference (times):", np.count_nonzero(invC)/sum)
    for i in range(3):
        shape = ttInvC[i].shape
        ttInvC[i] = np.reshape(ttInvC[i], [shape[0], shape[1]*shape[2], shape[3]])

        shape = ttC[i].shape
        ttC[i] = np.reshape(ttC[i], [shape[0], shape[1] * shape[2], shape[3]])

    newCshape = np.array(elementU.approxOrder - 2, dtype=int)
    invC = np.reshape(invC, [*newCshape, *newCshape])
    invC = np.transpose(invC, axes=[0, 3, 1, 4, 2, 5])
    invC = np.reshape(invC, newCshape**2)
    A = np.reshape(ttInvC[0], [ttInvC[0].shape[1], ttInvC[0].shape[-1]])
    initShape = invC.shape
    for i in range(1, len(ttInvC)):
        shape = ttInvC[i].shape
        reshaped = np.reshape(ttInvC[i], [shape[0], shape[1] * shape[2]])
        A = np.dot(A, reshaped)
        A = np.reshape(A, [np.prod(initShape[:i + 1]), int(A.shape[-1]/initShape[i])])
    A = np.reshape(A, initShape)
    print("approximation error for inverse: ", np.sum((A - invC)**2))

    newCshape = np.array(elementU.approxOrder - 2, dtype=int)
    C = np.reshape(C, [*newCshape, *newCshape])
    C = np.transpose(C, axes=[0, 3, 1, 4, 2, 5])
    C = np.reshape(C, newCshape ** 2)
    A = np.reshape(ttC[0], [ttC[0].shape[1], ttC[0].shape[-1]])
    initShape = C.shape
    for i in range(1, len(ttC)):
        shape = ttC[i].shape
        reshaped = np.reshape(ttC[i], [shape[0], shape[1] * shape[2]])
        A = np.dot(A, reshaped)
        A = np.reshape(A, [np.prod(initShape[:i + 1]), int(A.shape[-1] / initShape[i])])
    A = np.reshape(A, initShape)
    print("approximation error for original matrix: ", np.sum((A - C) ** 2))

elem = element(np.array([[-2, 2], [-4, 4], [-8, 8]]),
               7*np.array([2, 3, 4], dtype=int), np.array([0, 0, 0]))
def f(x):
    return np.exp(-np.sqrt(4*x[0]**2 + 2*x[1]**2 + x[2]**2))

# testKronSum(elem)
# testTT_approximation(elem, f, 1e-6)
testLaplacianInverseTT(elem)
