def main():
    # basisFuncList = []
    # basisDiffFuncList = []
    # t = time.time()
    # for i in range(elementU.getDim()):
    #     basisFuncList.append(elementU[i].evalAtChebPoints()[1:-1, 1: -1])
    #     basisDiffFuncList.append(elementU[i].evalDiffAtChebPoints()[1:-1, 1: -1])
    # C = np.kron(np.kron(np.kron(basisDiffFuncList[0], basisFuncList[1]), basisFuncList[2]), basisFuncList[3])
    # C += np.kron(np.kron(np.kron(basisFuncList[0], basisDiffFuncList[1]), basisFuncList[2]), basisFuncList[3])
    # C += np.kron(np.kron(np.kron(basisFuncList[0], basisFuncList[1]), basisDiffFuncList[2]), basisFuncList[3])
    # C += np.kron(np.kron(np.kron(basisFuncList[0], basisFuncList[1]), basisFuncList[2]), basisDiffFuncList[3])
    # print(time.time() - t)
    # print(C.shape)
    # old_C = C.copy()
    # print(C.shape)
    # C = np.reshape(C, np.hstack((elementU.approxOrder - 2, elementU.approxOrder - 2)))
    # C = np.transpose(C, axes=[0, 3, 1, 4, 2, 5])
    # C = np.transpose(C, axes=[0, 4, 1, 5, 2, 6, 3, 7])
    # C = np.reshape(C, (elementU.approxOrder - 2)**2)
    # C_TT_1 = approx.kronSumtoTT(basisFuncList, basisDiffFuncList)
    # print(C.shape)
    # approx.matrixTTsvd(old_C, shape=elementU.approxOrder - 2)
    # C_TT_2 = approx.simpleTTsvd(C, tol=1e-6, R_MAX=100)
    # for i in range(3):
    #     print(C_TT_1[i].shape)
    #     shape = C_TT_1[i].shape
    #     C_TT_1[i] = np.reshape(C_TT_1[i], [shape[0], shape[1]*shape[2], shape[3]])
    # if i == 0:
    #     print(i, "s core, upper")
    #     print(np.reshape(C_TT_2[i][0, :, 0], [3, 3]))
    #     print(i, "s core lower")
    #     print(np.reshape(C_TT_2[i][0, :, 1], [3, 3]))
    # print(C_TT_1[i].shape)
    # print(C_TT_2[i].shape)
    # print(np.max(C_TT_1[i] - C_TT_2[i]))
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
    grid = elementU.getGrid()
    weightArr = weight(grid)
    # weightArrTT = approx.simpleTTsvd(weightArr, tol=1e-2)
    # # weightArrTT = approx.vectorTTsvd(weightArr, TT_Tolerance)
    # # weightArrTT = approx.vecRound(weightArrTT, 1e-6)
    # print("before TT ", np.prod(weightArr.shape))
    # sum = 0
    # for it in weightArrTT:
    #     # print(it.shape, np.prod(it.shape))
    #     sum += np.prod(it.shape)
    # A = np.reshape(weightArrTT[0], [weightArrTT[0].shape[1], weightArrTT[0].shape[-1]])
    # initShape = weightArr.shape
    # for i in range(1, len(weightArrTT)):
    #     shape = weightArrTT[i].shape
    #     reshaped = np.reshape(weightArrTT[i], [shape[0], shape[1] * shape[2]])
    #     A = np.dot(A, reshaped)
    #     A = np.reshape(A, [np.prod(initShape[:i + 1]), int(A.shape[-1]/initShape[i])])
    # A = np.reshape(A, initShape)
    # print(np.sum((A - weightArr)**2))
    # #
    # # #
    # print("after TT ", sum)
    from itertools import permutations
    # a = np.arange(12)
    # perms = np.array(list(set(permutations(a))))
    def permutations(n):
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

    # perms = permutations(11)
    # print(perms.shape)
    # time.sleep(500)
    # arg_min = None
    # size_min = 100000
    # print(perms.shape[0])

    def f(x):
        perms = x
        binWeightArrTT = approx.simpleQTTsvd(weightArr, perms, 1e-6)
        sum = 0
        for it in binWeightArrTT:
            # print(it.shape, np.prod(it.shape))
            sum += np.prod(it.shape)

        # A = np.reshape(binWeightArrTT[0], [2, binWeightArrTT[0].shape[-1]])

        # initShape = weightArr.shape
        # for i in range(1, len(binWeightArrTT)):
        #     shape = binWeightArrTT[i].shape
        #     # print(shape)
        #     reshaped = np.reshape(binWeightArrTT[i], [shape[0], shape[1] * shape[2]])
        #     # print(A.shape, reshaped.shape)
        #     A = np.dot(A, reshaped)
        #     # print(A.shape)
        #     A = np.reshape(A, [2**(i + 1), int(A.shape[-1] / 2)])
        # A = np.reshape(A, initShape)
        # print(np.sum(np.abs(A - weightArr)))
        # print("after binaryTT ", sum)
        # print()
        # if sum <= size_min:
        #     size_min = sum
        #     arg_min = i
        #     print(i, sum)
        # print(sum)

    f(np.arange(12))
