from FiniteElementMethod.element.mainElementClass import element
import numpy as np
import misc.approximate as approx
def integrateBilinearForm0(elementU: element, integrationPointsAmount: int, TT_Tolerance, weight):
    """Integrates bilinear form of the type a(u, u) = int_K weight(x) u(x) v(x) dx,
        where U(x) and v(x) are basis functions of elementU element
        and K is a non-zero region of elementU functions.
        weight(x) must be a calculable function on an element grid

        Arguments:
            elementU:
            weight:
            integrationPointsAmount:
            TT_Tolerance:
        Returns:
            result: an integral of chosen bilinear form and elementU
    """

    basisFuncList = []
    basisDiffFuncList = []
    for i in range(elementU.getDim()):
        basisFuncList.append(elementU[i].evalAtChebPoints()[1:-1, 1: -1])
        basisDiffFuncList.append(elementU[i].evalDiffAtChebPoints()[1:-1, 1: -1])
    C = np.kron(np.kron(basisDiffFuncList[0], basisFuncList[1]), basisFuncList[2])
    C += np.kron(np.kron(basisFuncList[0], basisDiffFuncList[1]), basisFuncList[2])
    C += np.kron(np.kron(basisFuncList[0], basisFuncList[1]), basisDiffFuncList[2])
    old_C = C.copy()
    # print(C.shape)
    C = np.reshape(C, np.hstack((elementU.approxOrder - 2, elementU.approxOrder - 2)))
    C = np.transpose(C, axes=[0, 3, 1, 4, 2, 5])
    C = np.reshape(C, (elementU.approxOrder - 2)**2)
    C_TT_1 = approx.kronSumtoTT(basisFuncList, basisDiffFuncList)
    print(C.shape)
    # approx.matrixTTsvd(old_C, shape=elementU.approxOrder - 2)
    C_TT_2 = approx.simpleTTsvd(C, tol=1e-6, R_MAX=100)
    for i in range(3):
        # print(shape)
        shape = C_TT_1[i].shape
        C_TT_1[i] = np.reshape(C_TT_1[i], [shape[0], shape[1]*shape[2], shape[3]])
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

    A = np.reshape(C_TT_1[0], [C_TT_1[0].shape[1], C_TT_1[0].shape[-1]])
    initShape = C.shape
    for i in range(1, len(C_TT_1)):
        shape = C_TT_1[i].shape
        reshaped = np.reshape(C_TT_1[i], [shape[0], shape[1] * shape[2]])
        A = np.dot(A, reshaped)
        A = np.reshape(A, [np.prod(initShape[:i + 1]), int(A.shape[-1] / initShape[i])])
    A = np.reshape(A, initShape)
    print("second", np.sum((A - C) ** 2))
    C = np.reshape(C, old_C.shape)
    print("just for test", np.max(C - old_C))
    #
    # approx.kronSumtoTT()
    # grid = elementU.getGrid()
    # weightArr = weight(grid)
    # weightArrTT = approx.simpleTTsvd(weightArr, tol=1e-6)
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
    #     # print(reshaped.shape)
    #     A = np.dot(A, reshaped)
    #     # print(A.shape)
    #     # print(np.prod(initShape[:i+1]))
    #     # print(A.shape[-1], initShape[i+1])
    #     A = np.reshape(A, [np.prod(initShape[:i + 1]), int(A.shape[-1]/initShape[i])])
    # A = np.reshape(A, initShape)
    # print(np.sum((A - weightArr)**2))
    #
    # #
    # print("after TT ", sum)
    # # binWeightArrTT = approx.simpleQTTsvd(weightArr, 1e-6)
    # # # binWeightArrTT = approx.binaryVectorTTsvd(weightArr, 1e-6)
    # # # binWeightArrTT = approx.vecRound(binWeightArrTT, 1e-6)
    # # sum = 0
    # # for it in binWeightArrTT:
    # #     # print(it.shape, np.prod(it.shape))
    # #     sum += np.prod(it.shape)
    # #
    # # A = np.reshape(binWeightArrTT[0], [2, binWeightArrTT[0].shape[-1]])
    # #
    # # initShape = weightArr.shape
    # # for i in range(1, len(binWeightArrTT)):
    # #     shape = binWeightArrTT[i].shape
    # #     # print(shape)
    # #     reshaped = np.reshape(binWeightArrTT[i], [shape[0], shape[1] * shape[2]])
    # #     # print(A.shape, reshaped.shape)
    # #     A = np.dot(A, reshaped)
    # #     # print(A.shape)
    # #     A = np.reshape(A, [2**(i + 1), int(A.shape[-1] / 2)])
    # # A = np.reshape(A, initShape)
    # # print(np.sum(np.abs(A - weightArr)))
    # # print("after binaryTT ", sum)
    return None


def integrateBilinearForm1(elementU: element, weight, integrationPointsAmount: int):
    """(one-dimensional) Integrates bilinear form of the type a(u, u) = int_K weight(x) du(x) dv(x) dx,
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

    # def calc_func1d(self, f, lims, grid):
    #
    #     points = np.array(grid)
    #     res = np.zeros(points.size)
    #     lims = lims.T
    #     for i in range(lims.shape[0]):
    #         inidx = np.asarray(np.where(np.logical_and(lims[i, 0] <= points,
    #                                        points <= lims[i, 1])))
    #         if inidx.size > 0:
    #             inbox = points[inidx]
    #             F = (f(i, inbox)).flatten()
    #             res[inidx] = F
    #     res = res.flatten()
    #     return res
    #
    # def inner(self, g1, g2): #scalar product of 2 d-dim vectors. Returns list [[g1_1*g2_1, ..., g1_d*g2_d]]
    #     res = []
    #
    #     if isinstance(g1[0], list):
    #         for i in range(len(g1)):
    #             tmp_res = []
    #             for j in range(len(g1[i])):
    #                 lhs = g1[i][j]
    #                 rhs = g2[i][j]
    #                 tmp = lhs[:, :, None]*rhs[:, None, :]
    #                 x, y, z = tmp.shape
    #                 tmp = np.reshape(tmp, [x, y*z])
    #                 tmp_res.append(tmp)
    #             res.append(tmp_res)
    #     else:
    #
    #             tmp_res = []
    #             for j in range(len(g1)):
    #                 lhs = g1[j]
    #                 rhs = g2[j]
    #                 tmp = lhs[:, :, None]*rhs[:, None, :]
    #                 x, y, z = tmp.shape
    #                 tmp = np.reshape(tmp, [x, y*z])
    #                 tmp_res.append(tmp)
    #             res.append(tmp_res)
    #     return res

    def elementPartialDerivative(self, element: element, axis):
        """Calculates partial derivative of element basis functions along axis.

        Arguments:
        element: object of FiniteElementMethod/element/mainElementClass.py element class
        axis: number of axis/dimension over which differentiation is to be performed

        Returns:
        """
        basisFunctions = lambda x: element.ev
        basisFunctionDerivativeAlongAxis = lambda x: element.d(x)
        return None

    def grad1D(self, element: element, axis):

        basisFunctions = lambda x: element(x)
        basisFunctionDerivatives = lambda x: element.d(x)

        def res(x):
            dsx = basisFunctionDerivatives(x)
            fsx = basisFunctions(x)
            arr = []
            for i in range(len(elem.dims)):
                tmp = []
                for j in range(len(elem.dims)):
                    if i == j:
                        tmp.append(dsx[j])
                    else:
                        tmp.append(fsx[j])
                arr.append(tmp)
            return arr

        return res

    # def integr(self, K, elemF, F=None, n=300, infMap="linear"):
    #         # n=2000
    #         grids = []
    #         weights = []
    #         elem_grids = []
    #         ##iterating over all
    #         for i in range(K.shape[0]):
    #             if K[i, 0] != K[i, 1]:
    #                     tmp = b_elem(K[i], 2, infMap=infMap)
    #                     # tmp = b_elem(K[i], 2, infMap="linear")
    #                     w, x = intg.reg_32_wn(-1.0, 1.0, n)
    #                     w *= tmp.idmap(x)
    #                     w = np.array(w, dtype=np.float)
    #                     elem_grids.append(x)
    #                     grids.append(tmp.map(x))
    #                     weights.append(w)
    #             else:
    #                 elem_grids.append(K[i, 0])
    #                 grids.append(K[i, 0])
    #                 weights.append(np.ones(1))
    #
    #         fx = elemF(elem_grids)
    #         msh = approx.meshgrid(*grids)
    #         if K.shape[0] > 1:
    #             if F is not None:
    #                 f = F(*msh)
    #                 u = approx.vectorTTsvd(f, tol=1e-7)
    #
    #                 for i in range(len(u)):
    #                     u[i] *= weights[i][None, :, None]
    #             else:
    #                 u = []
    #                 for i in range(K.shape[0]):
    #                     u.append(weights[i][None, :, None])
    #             v = 0
    #             for it in fx:
    #                     v += approx.contraction(u, it)
    #         else:
    #             if F is not None:
    #                 f = F(np.array(*msh, dtype=np.float))
    #             else:
    #                 f = 1
    #             v = 0
    #             for it in fx:
    #                 v += np.sum(f*it[0].T*weights[0], axis=1)
    #         # print(v)
    #         # import time as time
    #         # time.sleep(500)
    #         return v
    #
    #
    # def integr2(self, K, elemF, F=None, n=10):
    #         n = 200
    #         grids = []
    #         weights = []
    #         c_weights = []
    #         elem_grids = []
    #         for i in range(K.shape[0]):
    #             if K[i, 0] != K[i, 1]:
    #                     tmp = b_elem(K[i], 2)
    #                     # w, x = intg.reg_32_wn(-1.0, 1.0, n)
    #                     cw, x = intg.clenshaw_wn(n)
    #                     w = tmp.idmap(x)
    #                     # if(np.size(w) > 1):
    #                     #     print(w[-3::])
    #                     elem_grids.append(x)
    #                     grids.append(tmp.map(x))
    #                     weights.append(w)
    #                     c_weights.append(cw)
    #             else:
    #                 elem_grids.append(K[i, 0])
    #                 grids.append(K[i, 0])
    #                 weights.append(np.ones(1))
    #
    #         fx = elemF(elem_grids)
    #
    #         # print('fx')
    #         # print(fx)
    #         msh = approx.meshgrid(*grids)
    #         if K.shape[0] > 1:
    #             if F is not None:
    #                 f = F(*msh)
    #                 u = approx.vectorTTsvd(f, tol=1e-7)
    #
    #                 for i in range(len(u)):
    #                     u[i] *= weights[i][None, :, None]
    #             else:
    #                 u = []
    #                 for i in range(K.shape[0]):
    #                     u.append(weights[i][None, :, None])
    #             v = 0
    #             for it in fx:
    #                     v += approx.contraction(u, it)
    #         else:
    #             if F is not None:
    #                 f = F(np.array(*msh, dtype=np.float))
    #             else:
    #                 f = 1
    #             v = 0
    #             for it in fx:
    #                 # v += np.sum(f*it[0].T*weights[0], axis=1)
    #                 if(np.size(f) == 1):
    #                     v += np.sum(f*it[0].T*weights[0], axis=1)
    #                 else:
    #                     res = f*it[0].T*weights[0]
    #                     # print('heey')
    #                     # print(res)
    #                     res = np.nan_to_num(res)
    #                     # print(res)
    #                     cf = spect.chebTransform(res.T)
    #                     v += np.sum(cf.T*c_weights[0], axis=1)
    #         # print(v)
    #         # import time as time
    #         # time.sleep(500)
    #         return v
    #
    #
    # def orth(self, g, K1, K2):
    #     meshObj = mesh(K1.shape[0])
    #     b = meshObj.intersection(K1, K2)
    #     gg = g.copy()
    #
    #     for i in range(b.shape[0]):
    #         if b[i, 0] == b[i, 1]:
    #             k = i
    #             if np.mean(K1[k, :]) > b[i, 0]:
    #                 gg[k][k] *= -1
    #             return gg[k]
    #
    # # def calc_func2d(self, f, lims, grid):
    # #
    # #     tmp_grid = np.array(grid)
    # #     points = np.reshape(tmp_grid, [2, tmp_grid.shape[1]*tmp_grid.shape[2]]).T
    # #     res = np.zeros(tmp_grid.shape[1]*tmp_grid.shape[2])
    # #     for i in range(len(lims)):
    # #         inidx = np.all(np.logical_and(lims[i][0] <= points,
    # #                                       points <= lims[i][1]), axis=1)
    # #
    # #         inbox = points[inidx]
    # #
    # #         xx = np.unique(inbox[:, 0])
    # #         yy = np.unique(inbox[:, 1])
    # #
    # #         F = (f(i, xx, yy).T).flatten()
    # #
    # #         res[inidx] = F
    # #     res = np.reshape(res, [tmp_grid.shape[1], tmp_grid.shape[2]])
    # #
    # #     return res


elem = element(np.array([[-2, 2], [-4, 4], [-8, 8]]), np.array([6, 6, 6]), np.array([0, 0, 0]))
def f(x):
    return np.exp(-4*x[0]**2 - 2*x[1]**2 - x[2]**2)

integrateBilinearForm0(elem, 100, 1e-6, f)
