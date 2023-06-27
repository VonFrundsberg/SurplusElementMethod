from FiniteElementMethod.element.basicElement import *

class element():
    def __init__(self, rectangle, polynomialOrder, boundaryConditions=None, mappingType=0):
        self.axes = []
        dimensionality = len(polynomialOrder)
        # for i in range(len(n)):
        #     if bc[i] is not None:
        #         self.dims.append(b_elem(K[i, :], n[i], bc=bc[i], infMap=infMap))
        #     else:
        #         self.dims.append(b_elem(K[i, :], n[i], infMap=infMap))
    def get_K(self):
        return self.dims[0].I
    def get_n(self):
        return self.dims[0].n


    def p_eval(self, x, mul=1): ## evaluates element values at x, multiplied by mul
        res = []
        i = 0
        for it in self.dims:
            res.append(mul*it.p_eval(np.unique(x[i])))
            i += 1

        return res

    def dp_eval(self, x):
        res = []
        i = 0
        for it in self.dims:
            res.append(it.dp_eval(np.unique(x[i])))
            i += 1

        return res
    def bcs(self):
        lists = np.zeros(len(self.dims))
        pad = np.zeros([len(self.dims), 2], dtype=np.int)
        pad = pad.tolist()

        for i in range(len(self.dims)):

            lists[i] += len(self.dims[i].bc)
            for it in self.dims[i].bc:
                if it[0] == 0:
                    pad[i][0] = 1
                if it[0] == -1:
                    pad[i][1] = 1
        return lists, pad
    def grid(self):
        grid = []
        for i in range(len(self.dims)):
            grid.append(self.dims[i].xs())
        return grid

    def new_grid(self):
        grid = []
        for i in range(len(self.dims)):
            grid.append(self.dims[i].new_xs())
        return grid
    def gen_func(self):
        return lambda x: self.p_eval(x)
    def gen_funcd(self):
        return lambda x: self.dp_eval(x)

    def __call__(self, x):
            return self.p_eval(x)
    def d(self, x):
            return self.dp_eval(x)