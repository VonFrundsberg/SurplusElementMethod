import time

from FiniteElementMethod.element.mainElementClass import element
import numpy as np
import mathematics.approximate as approx
from scipy.optimize import differential_evolution
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


elem = element(np.array([[-2, 2], [-4, 4], [-8, 8]]),
               2**3*np.array([2, 2, 2], dtype=int), np.array([0, 0, 0]))
def f(x):
    return np.exp(-np.sqrt(4*x[0]**2 + 2*x[1]**2 + x[2]**2))

integrateBilinearForm0(elem, 100, 1e-6, f)
