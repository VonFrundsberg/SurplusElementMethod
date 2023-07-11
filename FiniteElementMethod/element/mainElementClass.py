import numpy as np

from FiniteElementMethod.element.basicElement import basicElement
import mathematics.approximate as approx

class element():
    def __init__(self, rectangle, polynomialOrder, mappingType=0, boundaryConditions=None):
        self.axes = []
        self.approxOrder = np.atleast_1d(polynomialOrder)
        self.rectangle = np.atleast_2d(rectangle)
        self.dim = len(self.approxOrder)
        self.basicElements = []
        for i in range(self.dim):
        #     if bc[i] is not None:
        #         self.dims.append(b_elem(K[i, :], n[i], bc=bc[i], infMap=infMap))
        #     else:
            self.basicElements.append(basicElement(self.rectangle[i, :], self.approxOrder[i], mappingType[i]))

    def getDim(self):
        return self.dim
    def evaluatePoints(self, x):
        """ Evaluates basis functions at points x

                Arguments:
                    x: evaluation points

                Returns:

        """
        result = []
        i = 0
        for it in self.basicElements:
            result.append(it.eval(np.unique(x[i])))
            i += 1
        return result

    def evalDiffAlongAxis(self, x, axis):
        result = self.basicElements[axis].evalDiff(np.unique(x[axis]))
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
    def getGrid(self):
        coordsList = []
        for i in range(self.dim):
            coordsList.append(self.basicElements[i].getMappedRefPoints())
        grid = approx.meshgrid(*coordsList)
        return grid

    def __getitem__(self, key):
        return self.basicElements[key]

    # def gen_func(self):
    #     return lambda x: self.p_eval(x)
    # def gen_funcd(self):
    #     return lambda x: self.dp_eval(x)

    # def __call__(self, x):
    #         return self.p_eval(x)
    # def d(self, x):
    #         return self.dp_eval(x)


