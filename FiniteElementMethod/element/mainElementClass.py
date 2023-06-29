import numpy as np

from FiniteElementMethod.element.basicElement import basicElement

class element():
    def __init__(self, rectangle, polynomialOrder, mappingType=0, boundaryConditions=None):
        self.axes = []
        polynomialOrder = np.atleast_1d(polynomialOrder)
        rectangle = np.atleast_2d(rectangle)
        dimensionality = len(polynomialOrder)
        self.basicElements = []
        for i in range(dimensionality):
        #     if bc[i] is not None:
        #         self.dims.append(b_elem(K[i, :], n[i], bc=bc[i], infMap=infMap))
        #     else:
                self.basicElements.append(basicElement(rectangle[i, :], polynomialOrder[i], mappingType))
    # def get_K(self):
    #     return self.dims[0].I
    # def get_n(self):
    #     return self.dims[0].n


    def evaluatePoints(self, x):
        """ Evaluates basis functions at points x

                Arguments:
                    x: evaluation points

                Returns:

        """
        result = []
        i = 0
        for it in self.basicElements:
            result.append(it.evaluatePoints(np.unique(x[i])))
            i += 1
        return result

    def evaluateDerivativePointsAxis(self, x, axis):
        result = self.basicElements[axis].evaluateDerivativePoints(np.unique(x[axis]))
        return result
    # def bcs(self):
    #     lists = np.zeros(len(self.dims))
    #     pad = np.zeros([len(self.dims), 2], dtype=np.int)
    #     pad = pad.tolist()
    #
    #     for i in range(len(self.dims)):
    #
    #         lists[i] += len(self.dims[i].bc)
    #         for it in self.dims[i].bc:
    #             if it[0] == 0:
    #                 pad[i][0] = 1
    #             if it[0] == -1:
    #                 pad[i][1] = 1
    #     return lists, pad
    # def grid(self):
    #     grid = []
    #     for i in range(len(self.dims)):
    #         grid.append(self.dims[i].getMappedReferencePoins())
    #     return grid

    # def new_grid(self):
    #     grid = []
    #     for i in range(len(self.dims)):
    #         grid.append(self.dims[i].new_xs())
    #     return grid
    # def gen_func(self):
    #     return lambda x: self.p_eval(x)
    # def gen_funcd(self):
    #     return lambda x: self.dp_eval(x)

    # def __call__(self, x):
    #         return self.p_eval(x)
    # def d(self, x):
    #         return self.dp_eval(x)

elem = element(np.array([0, 1]), 2, 0)