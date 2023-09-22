
import numpy as np
import scipy.linalg as sp_linalg
import mathematics.approximate as approx
from scikit_tt.tensor_train import TT
import matplotlib.pyplot as plt
def expandCoreCase0(argTensor, index: int):
    """
    case when there is a just a product of two 1, 1 cores in which first is packed with 2 dims
    """
    #printTT(argTensor)
    n = 3;
    #lhsLeft = [None]*n; lhsRight = [None]*n;
    lhs = [None]*n
    rand2 = np.random.rand(2, 2); rand4 = np.random.rand(4, 4)
    lhs = np.einsum("i, j -> ij", rand2.flatten(), rand4.flatten())
    lhsKron = np.kron(rand2, rand4)
    for i in range(1, n):
        rand2 = np.random.rand(2, 2)
        rand4 = np.random.rand(4, 4)

        lhs += np.einsum("i, j -> ij", rand2.flatten(), rand4.flatten())
        lhsKron += np.kron(rand2, rand4)
    #print(lhs.shape)

    lhs = np.array(lhs)[np.newaxis, :, :, np.newaxis]
    rhs = np.random.rand(3, 3)[np.newaxis, :, :, np.newaxis]
    cores = [lhsKron[np.newaxis, :, :, np.newaxis], rhs]
    print("init sort of cores")
    approx.printTT([lhs, rhs])
    ttCores = TT(cores)
    fullTensor = ttCores.full()
    print("initial kronTensor shape", fullTensor.shape)
    lhsShape = lhs.shape
    #reshapedLhs = np.transpose(lhs, (0, 1, 3, 2))
    #print("prev shape", lhs.shape)
    reshapedLhs = np.reshape(lhs, [4, 16])
    #print("after shape", reshapedLhs.shape)
    u, s, v = sp_linalg.svd(reshapedLhs, full_matrices=False)
    #print(s[:30])
    cumsum = np.cumsum(s)
    r_delta = np.argmax(cumsum[-1] - cumsum < 1e-12) + 1
    #r_delta = 1000
    u = np.dot(u[:, :r_delta], np.diag(np.sqrt(s[:r_delta])))
    v = np.dot(np.diag(np.sqrt(s[:r_delta])), v[:r_delta, :])
    print("separated shapes")
    print(u.shape, v.shape)
    sqrtU = int(np.sqrt(u.shape[0]))
    u = np.reshape(u, [1, sqrtU, sqrtU, u.shape[1]])
    #u = np.transpose(u, (0, 2, 1, 3))
    sqrtV = int(np.sqrt(v.shape[1]))
    v = np.reshape(v, [v.shape[0], sqrtV, sqrtV, 1])
    #v = np.transpose(v, (0, 2, 1, 3))
    # prod = np.einsum("ijkl, lmnv -> ijkmnv", u, v)
    # prod = np.reshape(prod, [1, 4, 4, 1])
    #
    # print(np.max(np.abs(lhs - prod)))
    # cores = np.reshape(cores, [2, 2, 3, 2, 2, 3])
    newTensor = TT([u, v, rhs])
    print("cores of newTensor")
    approx.printTT(newTensor.cores)
    print("newTensor shape")
    print(newTensor.full().shape)
    # print(toFullTensor([u, v, rhs], matrixForm=True).shape)
    # plt.scatter(np.arange(newTensor.full().size),
    #             toFullTensor([u, v, rhs], matrixForm=False).flatten() -
    #             toFullTensor([lhs, rhs], matrixForm=False).flatten())
    # plt.show()

    plt.scatter(np.arange(fullTensor.size),
               approx.toFullTensor([u, v, rhs], matrixForm=True).flatten() - fullTensor.flatten())
    plt.show()
    #print(newTensor.full().flatten()[n:n + 10] - fullTensor.flatten()[n:n + 10])

    pass

def expandCoreCase1(argTensor, index: int):
    """
        case where there is a sum of size alpha of two cores product
        in which first core is packed with 2 dims
        """
    n = 3;
    alpha = 3
    #lhsLeft = [None]*n; lhsRight = [None]*n;
    lhs = [None]*n
    rand2 = np.random.rand(2, 2); rand4 = np.random.rand(4, 4)
    randAlpha = np.random.rand(alpha)
    lhs = np.einsum("i, j, k -> ijk", rand2.flatten(), rand4.flatten(), randAlpha)
    lhsKron = np.einsum("ij, k -> ijk", np.kron(rand2, rand4), randAlpha)
    for i in range(1, n):
        rand2 = np.random.rand(2, 2)
        rand4 = np.random.rand(4, 4)

        lhs += np.einsum("i, j, k -> ijk", rand2.flatten(), rand4.flatten(), randAlpha)
        lhsKron += np.einsum("ij, k -> ijk", np.kron(rand2, rand4), randAlpha)
    #print(lhs.shape)
    lhsKron = lhsKron[np.newaxis, :, :, :]
    lhs = np.array(lhs)[np.newaxis, :, :]
    rhs = np.random.rand(alpha, 3, 3)[:, :, :, np.newaxis]
    #print(lhsKron.shape)
    cores = [lhsKron, rhs]
    print("init sort of cores")
    approx.printTT([lhs, rhs])
    print("init kronCores")
    approx.rintTT([lhsKron, rhs])
    ttCores = TT(cores)
    fullTensor = ttCores.full()
    print("initial kronTensor shape", fullTensor.shape)
    lhsShape = lhs.shape
    #reshapedLhs = np.transpose(lhs, (0, 1, 3, 2))
    #print("prev shape", lhs.shape)
    reshapedLhs = np.reshape(lhs, [4, 16*alpha])
    #print("after shape", reshapedLhs.shape)
    u, s, v = sp_linalg.svd(reshapedLhs, full_matrices=False)
    #print(s[:30])
    cumsum = np.cumsum(s)
    r_delta = np.argmax(cumsum[-1] - cumsum < 1e-12) + 1
    #r_delta = 1000
    u = np.dot(u[:, :r_delta], np.diag(np.sqrt(s[:r_delta])))
    v = np.dot(np.diag(np.sqrt(s[:r_delta])), v[:r_delta, :])
    print("separated shapes")
    print(u.shape, v.shape)
    sqrtU = int(np.sqrt(4))
    u = np.reshape(u, [1, sqrtU, sqrtU, u.shape[1]])
    #u = np.transpose(u, (0, 2, 1, 3))
    sqrtV = int(np.sqrt(16))
    v = np.reshape(v, [v.shape[0], sqrtV, sqrtV, alpha])
    #v = np.transpose(v, (0, 2, 1, 3))
    # prod = np.einsum("ijkl, lmnv -> ijkmnv", u, v)
    # prod = np.reshape(prod, [1, 4, 4, 1])
    #
    # print(np.max(np.abs(lhs - prod)))
    # cores = np.reshape(cores, [2, 2, 3, 2, 2, 3])
    newTensor = TT([u, v, rhs])
    print("cores of newTensor")
    approx.printTT(newTensor.cores)
    print("newTensor shape")
    print(newTensor.full().shape)
    # print(toFullTensor([u, v, rhs], matrixForm=True).shape)
    # plt.scatter(np.arange(newTensor.full().size),
    #             toFullTensor([u, v, rhs], matrixForm=False).flatten() -
    #             toFullTensor([lhs, rhs], matrixForm=False).flatten())
    # plt.show()

    plt.scatter(np.arange(fullTensor.size),
               approx.toFullTensor([u, v, rhs], matrixForm=True).flatten() - fullTensor.flatten())
    plt.show()
    #print(newTensor.full().flatten()[n:n + 10] - fullTensor.flatten()[n:n + 10])

    pass


def expandCoreCase2(argTensor, index: int):
    """
        case where there is a sum of size alpha of two cores product
        in which both cores are packed with 2 dims
        """

    n = 4;
    alpha = 3

    lhs = [None] * n
    rand2 = np.random.rand(2, 2);
    rand4 = np.random.rand(4, 4)
    randAlpha = np.random.rand(alpha)
    lhs = np.einsum("i, j, k -> ijk", rand2.flatten(), rand4.flatten(), randAlpha)
    lhsKron = np.einsum("ij, k -> ijk", np.kron(rand2, rand4), randAlpha)
    for i in range(1, n):
        rand2 = np.random.rand(2, 2)
        rand4 = np.random.rand(4, 4)

        lhs += np.einsum("i, j, k -> ijk", rand2.flatten(), rand4.flatten(), randAlpha)
        lhsKron += np.einsum("ij, k -> ijk", np.kron(rand2, rand4), randAlpha)

    lhsKron = lhsKron[np.newaxis, :, :, :]
    lhs = np.array(lhs)[np.newaxis, :, :]

    rhs = [None] * n
    rand2 = np.random.rand(3, 3);
    rand4 = np.random.rand(5, 5)
    randAlpha = np.random.rand(alpha)
    rhs = np.einsum("i, j, k -> ijk", randAlpha, rand2.flatten(), rand4.flatten())
    rhsKron = np.einsum("k, ij -> kij", randAlpha, np.kron(rand2, rand4))
    for i in range(1, n):
        rand2 = np.random.rand(3, 3);
        rand4 = np.random.rand(5, 5)

        rhs += np.einsum("i, j, k -> ijk", randAlpha, rand2.flatten(), rand4.flatten())
        rhsKron += np.einsum("k, ij -> kij", randAlpha, np.kron(rand2, rand4))

    rhsKron = rhsKron[:, :, :, np.newaxis]
    rhs = np.array(rhs)[:, :, :, np.newaxis]

    #rhs = np.random.rand(alpha, 3, 3)[:, :, :, np.newaxis]
    #print(lhsKron.shape)
    cores = [lhsKron, rhsKron]
    print("init sort of cores")
    approx.printTT([lhs, rhs])
    print("init kronCores")
    approx.printTT([lhsKron, rhsKron])
    ttCores = TT(cores)
    fullTensor = ttCores.full()
    print("initial kronTensor shape", fullTensor.shape)
    lhsShape = lhs.shape
    #reshapedLhs = np.transpose(lhs, (0, 1, 3, 2))
    #print("prev shape", lhs.shape)
    """
    separation of left core
    """
    reshapedLhs = np.reshape(lhs, [4, 16*alpha])
    #print("after shape", reshapedLhs.shape)
    u, s, v = sp_linalg.svd(reshapedLhs, full_matrices=False)
    #print(s[:30])
    cumsum = np.cumsum(s)
    r_delta = np.argmax(cumsum[-1] - cumsum < 1e-12) + 1
    #r_delta = 1000
    u = np.dot(u[:, :r_delta], np.diag(np.sqrt(s[:r_delta])))
    v = np.dot(np.diag(np.sqrt(s[:r_delta])), v[:r_delta, :])
    print("separated shapes at left core")
    print(u.shape, v.shape)
    sqrtU = int(np.sqrt(4))
    uLhs = np.reshape(u, [1, sqrtU, sqrtU, u.shape[1]])
    #u = np.transpose(u, (0, 2, 1, 3))
    sqrtV = int(np.sqrt(16))
    vLhs = np.reshape(v, [v.shape[0], sqrtV, sqrtV, alpha])
    """
        separation of right core
    """
    reshapedRhs = np.reshape(rhs, [alpha*9, 25])
    # print("after shape", reshapedLhs.shape)
    u, s, v = sp_linalg.svd(reshapedRhs, full_matrices=False)
    # print(s[:30])
    cumsum = np.cumsum(s)
    r_delta = np.argmax(cumsum[-1] - cumsum < 1e-12) + 1
    # r_delta = 1000
    u = np.dot(u[:, :r_delta], np.diag(np.sqrt(s[:r_delta])))
    v = np.dot(np.diag(np.sqrt(s[:r_delta])), v[:r_delta, :])
    print("separated shapes of right core")
    print(u.shape, v.shape)
    sqrtU = int(np.sqrt(9))
    uRhs = np.reshape(u, [alpha, sqrtU, sqrtU, u.shape[1]])
    # u = np.transpose(u, (0, 2, 1, 3))
    sqrtV = int(np.sqrt(25))
    vRhs = np.reshape(v, [v.shape[0], sqrtV, sqrtV, 1])
    print("resulting separated core shapes")
    approx.printTT([uLhs, vLhs, uRhs, vRhs])
    newTensor = TT([uLhs, vLhs, uRhs, vRhs])
    print("cores of newTensor")
    approx.printTT(newTensor.cores)
    print("newTensor shape")
    print(newTensor.full().shape)


    plt.scatter(np.arange(fullTensor.size),
               newTensor.full().flatten() - fullTensor.flatten())
    plt.show()


    pass

def expandCoreCase3(argTensor, index: int):
    """
        case where there is three cores with ranks alpha, beta
        in which first and last core are packed with 2 dims
        """
    n = 3;
    alpha = 3; beta = 4;

    lhs = [None] * n
    rand2 = np.random.rand(2, 2);
    rand4 = np.random.rand(4, 4)
    randAlpha = np.random.rand(alpha)
    lhs = np.einsum("i, j, k -> ijk", rand2.flatten(), rand4.flatten(), randAlpha)
    lhsKron = np.einsum("ij, k -> ijk", np.kron(rand2, rand4), randAlpha)
    for i in range(1, n):
        rand2 = np.random.rand(2, 2)
        rand4 = np.random.rand(4, 4)

        lhs += np.einsum("i, j, k -> ijk", rand2.flatten(), rand4.flatten(), randAlpha)
        lhsKron += np.einsum("ij, k -> ijk", np.kron(rand2, rand4), randAlpha)

    lhsKron = lhsKron[np.newaxis, :, :, :]
    lhs = np.array(lhs)[np.newaxis, :, :]

    rhs = [None] * n
    rand2 = np.random.rand(3, 3);
    rand4 = np.random.rand(5, 5)
    randBeta = np.random.rand(beta)
    rhs = np.einsum("i, j, k -> ijk", randBeta, rand2.flatten(), rand4.flatten())
    rhsKron = np.einsum("k, ij -> kij", randBeta, np.kron(rand2, rand4))
    for i in range(1, n):
        rand2 = np.random.rand(3, 3);
        rand4 = np.random.rand(5, 5)

        rhs += np.einsum("i, j, k -> ijk", randBeta, rand2.flatten(), rand4.flatten())
        rhsKron += np.einsum("k, ij -> kij", randBeta, np.kron(rand2, rand4))

    rhsKron = rhsKron[:, :, :, np.newaxis]
    rhs = np.array(rhs)[:, :, :, np.newaxis]

    mid = np.random.rand(alpha, 3, 3, beta)
    #print(lhsKron.shape)
    cores = [lhsKron, mid, rhsKron]
    print("init sort of cores")
    approx.printTT([lhs, mid, rhs])
    print("init kronCores")
    approx.printTT([lhsKron, mid, rhsKron])
    ttCores = TT(cores)
    fullTensor = ttCores.full()
    print("initial kronTensor shape", fullTensor.shape)
    lhsShape = lhs.shape
    #reshapedLhs = np.transpose(lhs, (0, 1, 3, 2))
    #print("prev shape", lhs.shape)
    """
    separation of left core
    """
    reshapedLhs = np.reshape(lhs, [4, 16*alpha])
    #print("after shape", reshapedLhs.shape)
    u, s, v = sp_linalg.svd(reshapedLhs, full_matrices=False)
    #print(s[:30])
    cumsum = np.cumsum(s)
    r_delta = np.argmax(cumsum[-1] - cumsum < 1e-12) + 1
    #r_delta = 1000
    u = np.dot(u[:, :r_delta], np.diag(np.sqrt(s[:r_delta])))
    v = np.dot(np.diag(np.sqrt(s[:r_delta])), v[:r_delta, :])
    print("separated shapes at left core")
    print(u.shape, v.shape)
    sqrtU = int(np.sqrt(4))
    uLhs = np.reshape(u, [1, sqrtU, sqrtU, u.shape[1]])
    #u = np.transpose(u, (0, 2, 1, 3))
    sqrtV = int(np.sqrt(16))
    vLhs = np.reshape(v, [v.shape[0], sqrtV, sqrtV, alpha])
    """
        separation of right core
    """
    reshapedRhs = np.reshape(rhs, [beta*9, 25])
    # print("after shape", reshapedLhs.shape)
    u, s, v = sp_linalg.svd(reshapedRhs, full_matrices=False)
    # print(s[:30])
    cumsum = np.cumsum(s)
    r_delta = np.argmax(cumsum[-1] - cumsum < 1e-12) + 1
    # r_delta = 1000
    u = np.dot(u[:, :r_delta], np.diag(np.sqrt(s[:r_delta])))
    v = np.dot(np.diag(np.sqrt(s[:r_delta])), v[:r_delta, :])
    print("separated shapes of right core")
    print(u.shape, v.shape)
    sqrtU = int(np.sqrt(9))
    uRhs = np.reshape(u, [beta, sqrtU, sqrtU, u.shape[1]])
    # u = np.transpose(u, (0, 2, 1, 3))
    sqrtV = int(np.sqrt(25))
    vRhs = np.reshape(v, [v.shape[0], sqrtV, sqrtV, 1])
    print("resulting separated core shapes")
    approx.printTT([uLhs, vLhs, mid, uRhs, vRhs])
    newTensor = TT([uLhs, vLhs, mid, uRhs, vRhs])
    print("cores of newTensor")
    approx.printTT(newTensor.cores)
    print("newTensor shape")
    print(newTensor.full().shape)


    plt.scatter(np.arange(fullTensor.size),
               newTensor.full().flatten() - fullTensor.flatten())
    plt.show()


    pass

def expandCoreCase4(argTensor, index: int):
    """
        case where there is three cores with ranks alpha, beta
        in which all cores are packed with 2 dims
        """
    n = 3;
    alpha = 3; beta = 4;

    lhs = [None] * n
    rand2 = np.random.rand(2, 2);
    rand4 = np.random.rand(4, 4)
    randAlpha = np.random.rand(alpha)
    lhs = np.einsum("i, j, k -> ijk", rand2.flatten(), rand4.flatten(), randAlpha)
    lhsKron = np.einsum("ij, k -> ijk", np.kron(rand2, rand4), randAlpha)
    for i in range(1, n):
        rand2 = np.random.rand(2, 2)
        rand4 = np.random.rand(4, 4)

        lhs += np.einsum("i, j, k -> ijk", rand2.flatten(), rand4.flatten(), randAlpha)
        lhsKron += np.einsum("ij, k -> ijk", np.kron(rand2, rand4), randAlpha)

    lhsKron = lhsKron[np.newaxis, :, :, :]
    lhs = np.array(lhs)[np.newaxis, :, :]

    rhs = [None] * n
    rand2 = np.random.rand(3, 3);
    rand4 = np.random.rand(5, 5)
    randBeta = np.random.rand(beta)
    rhs = np.einsum("i, j, k -> ijk", randBeta, rand2.flatten(), rand4.flatten())
    rhsKron = np.einsum("k, ij -> kij", randBeta, np.kron(rand2, rand4))
    for i in range(1, n):
        rand2 = np.random.rand(3, 3);
        rand4 = np.random.rand(5, 5)
        randBeta = np.random.rand(beta)

        rhs += np.einsum("i, j, k -> ijk", randBeta, rand2.flatten(), rand4.flatten())
        rhsKron += np.einsum("k, ij -> kij", randBeta, np.kron(rand2, rand4))

    rhsKron = rhsKron[:, :, :, np.newaxis]
    rhs = np.array(rhs)[:, :, :, np.newaxis]

    lhs = [None] * n
    rand2 = np.random.rand(2, 2);
    rand4 = np.random.rand(4, 4)
    randAlpha = np.random.rand(alpha)
    lhs = np.einsum("i, j, k -> ijk", rand2.flatten(), rand4.flatten(), randAlpha)
    lhsKron = np.einsum("ij, k -> ijk", np.kron(rand2, rand4), randAlpha)
    for i in range(1, n):
        rand2 = np.random.rand(2, 2)
        rand4 = np.random.rand(4, 4)
        randAlpha = np.random.rand(alpha)

        lhs += np.einsum("i, j, k -> ijk", rand2.flatten(), rand4.flatten(), randAlpha)
        lhsKron += np.einsum("ij, k -> ijk", np.kron(rand2, rand4), randAlpha)

    lhsKron = lhsKron[np.newaxis, :, :, :]
    lhs = np.array(lhs)[np.newaxis, :, :]

    mid = [None] * n
    rand2 = np.random.rand(6, 6);
    rand4 = np.random.rand(7, 7)
    randAlpha = np.random.rand(alpha)
    randBeta = np.random.rand(beta)
    mid = np.einsum("i, j, k, m -> ijkm",
                    randAlpha, rand2.flatten(), rand4.flatten(), randBeta)
    midKron = np.einsum("k, ij, m -> kijm", randAlpha, np.kron(rand2, rand4), randBeta)
    for i in range(1, n):
        rand2 = np.random.rand(6, 6);
        rand4 = np.random.rand(7, 7)
        randAlpha = np.random.rand(alpha)
        randBeta = np.random.rand(beta)

        mid += np.einsum("i, j, k, m -> ijkm",
                    randAlpha, rand2.flatten(), rand4.flatten(), randBeta)
        midKron += np.einsum("k, ij, m -> kijm", randAlpha, np.kron(rand2, rand4), randBeta)

    #mid = np.random.rand(alpha, 3, 3, beta)
    #print(lhsKron.shape)
    cores = [lhsKron, midKron, rhsKron]
    print("init sort of cores")
    approx.printTT([lhs, mid, rhs])
    print("init kronCores")
    approx.printTT([lhsKron, midKron, rhsKron])
    ttCores = TT(cores)
    fullTensor = ttCores.full()
    print("initial kronTensor shape", fullTensor.shape)
    lhsShape = lhs.shape
    #reshapedLhs = np.transpose(lhs, (0, 1, 3, 2))
    #print("prev shape", lhs.shape)
    """
    separation of left core
    """
    reshapedLhs = np.reshape(lhs, [4, 16*alpha])
    #print("after shape", reshapedLhs.shape)
    u, s, v = sp_linalg.svd(reshapedLhs, full_matrices=False)
    #print(s[:30])
    cumsum = np.cumsum(s)
    r_delta = np.argmax(cumsum[-1] - cumsum < 1e-12) + 1
    #r_delta = 1000
    u = np.dot(u[:, :r_delta], np.diag(np.sqrt(s[:r_delta])))
    v = np.dot(np.diag(np.sqrt(s[:r_delta])), v[:r_delta, :])
    print("separated shapes at left core")
    print(u.shape, v.shape)
    sqrtU = int(np.sqrt(4))
    uLhs = np.reshape(u, [1, sqrtU, sqrtU, u.shape[1]])
    #u = np.transpose(u, (0, 2, 1, 3))
    sqrtV = int(np.sqrt(16))
    vLhs = np.reshape(v, [v.shape[0], sqrtV, sqrtV, alpha])
    """
        separation of right core
    """
    reshapedRhs = np.reshape(rhs, [beta*9, 25])
    # print("after shape", reshapedLhs.shape)
    u, s, v = sp_linalg.svd(reshapedRhs, full_matrices=False)
    # print(s[:30])
    cumsum = np.cumsum(s)
    r_delta = np.argmax(cumsum[-1] - cumsum < 1e-12) + 1
    # r_delta = 1000
    u = np.dot(u[:, :r_delta], np.diag(np.sqrt(s[:r_delta])))
    v = np.dot(np.diag(np.sqrt(s[:r_delta])), v[:r_delta, :])
    print("separated shapes of right core")
    print(u.shape, v.shape)
    sqrtU = int(np.sqrt(9))
    uRhs = np.reshape(u, [beta, sqrtU, sqrtU, u.shape[1]])
    # u = np.transpose(u, (0, 2, 1, 3))
    sqrtV = int(np.sqrt(25))
    vRhs = np.reshape(v, [v.shape[0], sqrtV, sqrtV, 1])

    """
        separation of mid core
    """
    reshapedMid = np.reshape(mid, [36*alpha, 49*beta])
    # print("after shape", reshapedLhs.shape)
    u, s, v = sp_linalg.svd(reshapedMid, full_matrices=False)
    # print(s[:30])
    # time.sleep(500)
    cumsum = np.cumsum(s)
    r_delta = np.argmax(cumsum[-1] - cumsum < 1e-12) + 1

    # r_delta = 1000
    u = np.dot(u[:, :r_delta], np.diag(np.sqrt(s[:r_delta])))
    v = np.dot(np.diag(np.sqrt(s[:r_delta])), v[:r_delta, :])
    print("separated shapes at mid core")
    print(u.shape, v.shape)
    sqrtU = int(np.sqrt(36))
    uMid = np.reshape(u, [alpha, sqrtU, sqrtU, u.shape[1]])
    # u = np.transpose(u, (0, 2, 1, 3))
    sqrtV = int(np.sqrt(49))
    vMid = np.reshape(v, [v.shape[0], sqrtV, sqrtV, beta])



    print("resulting separated core shapes")
    approx.printTT([uLhs, vLhs, uMid, vMid, uRhs, vRhs])
    newTensor = TT([uLhs, vLhs, uMid, vMid, uRhs, vRhs])
    print("cores of newTensor")
    approx.printTT(newTensor.cores)
    print("newTensor shape")
    print(newTensor.full().shape)

    print(np.max(np.abs(newTensor.full().flatten() - fullTensor.flatten())))
    # plt.scatter(np.arange(fullTensor.size),
    #            newTensor.full().flatten() - fullTensor.flatten())
    # plt.show()
    pass