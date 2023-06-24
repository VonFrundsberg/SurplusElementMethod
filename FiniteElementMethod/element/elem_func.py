import numpy as np
from FiniteElementMethod.element.basis_elem import *
from misc import integrate as intg
from misc import approximate as approx
from misc import spectral as spect
from FiniteElementMethod.mesh.mesh import *

class func():
    def __init__(self):
        return

    def calc_func1d(self, f, lims, grid):

        points = np.array(grid)
        res = np.zeros(points.size)
        lims = lims.T
        for i in range(lims.shape[0]):
            inidx = np.asarray(np.where(np.logical_and(lims[i, 0] <= points,
                                           points <= lims[i, 1])))
            if inidx.size > 0:
                inbox = points[inidx]
                F = (f(i, inbox)).flatten()
                res[inidx] = F
        res = res.flatten()
        return res

    def inner(self, g1, g2): #scalar product of 2 d-dim vectors. Returns list [[g1_1*g2_1, ..., g1_d*g2_d]]
        res = []

        if isinstance(g1[0], list):
            for i in range(len(g1)):
                tmp_res = []
                for j in range(len(g1[i])):
                    lhs = g1[i][j]
                    rhs = g2[i][j]
                    tmp = lhs[:, :, None]*rhs[:, None, :]
                    x, y, z = tmp.shape
                    tmp = np.reshape(tmp, [x, y*z])
                    tmp_res.append(tmp)
                res.append(tmp_res)
        else:

                tmp_res = []
                for j in range(len(g1)):
                    lhs = g1[j]
                    rhs = g2[j]
                    tmp = lhs[:, :, None]*rhs[:, None, :]
                    x, y, z = tmp.shape
                    tmp = np.reshape(tmp, [x, y*z])
                    tmp_res.append(tmp)
                res.append(tmp_res)
        return res

    def grad(self, elem):
        fs = lambda x: elem(x)
        ds = lambda x: elem.d(x)
        def res(x):
            dsx = ds(x)
            fsx = fs(x)
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


    def integr(self, K, elemF, F=None, n=300, infMap="linear"):
            # n=2000
            grids = []
            weights = []
            elem_grids = []
            ##iterating over all
            for i in range(K.shape[0]):
                if K[i, 0] != K[i, 1]:
                        tmp = b_elem(K[i], 2, infMap=infMap)
                        # tmp = b_elem(K[i], 2, infMap="linear")
                        w, x = intg.reg_32_wn(-1.0, 1.0, n)
                        w *= tmp.idmap(x)
                        w = np.array(w, dtype=np.float)
                        elem_grids.append(x)
                        grids.append(tmp.map(x))
                        weights.append(w)
                else:
                    elem_grids.append(K[i, 0])
                    grids.append(K[i, 0])
                    weights.append(np.ones(1))

            fx = elemF(elem_grids)
            msh = approx.meshgrid(*grids)
            if K.shape[0] > 1:
                if F is not None:
                    f = F(*msh)
                    u = approx.vectorTTsvd(f, tol=1e-7)

                    for i in range(len(u)):
                        u[i] *= weights[i][None, :, None]
                else:
                    u = []
                    for i in range(K.shape[0]):
                        u.append(weights[i][None, :, None])
                v = 0
                for it in fx:
                        v += approx.contraction(u, it)
            else:
                if F is not None:
                    f = F(np.array(*msh, dtype=np.float))
                else:
                    f = 1
                v = 0
                for it in fx:
                    v += np.sum(f*it[0].T*weights[0], axis=1)
            # print(v)
            # import time as time
            # time.sleep(500)
            return v


    def integr2(self, K, elemF, F=None, n=10):
            n = 200
            grids = []
            weights = []
            c_weights = []
            elem_grids = []
            for i in range(K.shape[0]):
                if K[i, 0] != K[i, 1]:
                        tmp = b_elem(K[i], 2)
                        # w, x = intg.reg_32_wn(-1.0, 1.0, n)
                        cw, x = intg.clenshaw_wn(n)
                        w = tmp.idmap(x)
                        # if(np.size(w) > 1):
                        #     print(w[-3::])
                        elem_grids.append(x)
                        grids.append(tmp.map(x))
                        weights.append(w)
                        c_weights.append(cw)
                else:
                    elem_grids.append(K[i, 0])
                    grids.append(K[i, 0])
                    weights.append(np.ones(1))

            fx = elemF(elem_grids)

            # print('fx')
            # print(fx)
            msh = approx.meshgrid(*grids)
            if K.shape[0] > 1:
                if F is not None:
                    f = F(*msh)
                    u = approx.vectorTTsvd(f, tol=1e-7)

                    for i in range(len(u)):
                        u[i] *= weights[i][None, :, None]
                else:
                    u = []
                    for i in range(K.shape[0]):
                        u.append(weights[i][None, :, None])
                v = 0
                for it in fx:
                        v += approx.contraction(u, it)
            else:
                if F is not None:
                    f = F(np.array(*msh, dtype=np.float))
                else:
                    f = 1
                v = 0
                for it in fx:
                    # v += np.sum(f*it[0].T*weights[0], axis=1)
                    if(np.size(f) == 1):
                        v += np.sum(f*it[0].T*weights[0], axis=1)
                    else:
                        res = f*it[0].T*weights[0]
                        # print('heey')
                        # print(res)
                        res = np.nan_to_num(res)
                        # print(res)
                        cf = spect.chebTransform(res.T)
                        v += np.sum(cf.T*c_weights[0], axis=1)
            # print(v)
            # import time as time
            # time.sleep(500)
            return v


    def orth(self, g, K1, K2):
        meshObj = mesh(K1.shape[0])
        b = meshObj.intersection(K1, K2)
        gg = g.copy()

        for i in range(b.shape[0]):
            if b[i, 0] == b[i, 1]:
                k = i
                if np.mean(K1[k, :]) > b[i, 0]:
                    gg[k][k] *= -1
                return gg[k]

    # def calc_func2d(self, f, lims, grid):
    #
    #     tmp_grid = np.array(grid)
    #     points = np.reshape(tmp_grid, [2, tmp_grid.shape[1]*tmp_grid.shape[2]]).T
    #     res = np.zeros(tmp_grid.shape[1]*tmp_grid.shape[2])
    #     for i in range(len(lims)):
    #         inidx = np.all(np.logical_and(lims[i][0] <= points,
    #                                       points <= lims[i][1]), axis=1)
    #
    #         inbox = points[inidx]
    #
    #         xx = np.unique(inbox[:, 0])
    #         yy = np.unique(inbox[:, 1])
    #
    #         F = (f(i, xx, yy).T).flatten()
    #
    #         res[inidx] = F
    #     res = np.reshape(res, [tmp_grid.shape[1], tmp_grid.shape[2]])
    #
    #     return res
