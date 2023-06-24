from misc import approximate
import itertools as itertools
from functools import reduce
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sympy as symp
import numpy as np
import re

class mesh():
    def __init__(self, n):
        self.n = n
        return
    # def gen_mesh(self, K, n=10, p=3, split=None):
    #     """Old function used for generating uniform rectangular mesh
    #     """
    #     n = np.atleast_1d(n)
    #     K = np.atleast_2d(K)
    #     x = []
    #     self.box = K.copy()
    #     if split is None:
    #         for i in range(K.shape[0]):
    #             x.append(np.linspace(K[i, 0], K[i, 1], n[i]))
    #     else:
    #         for i in range(K.shape[0]):
    #             x.append(split)
    #     l = []
    #     neigh = []
    #     ind = np.zeros(len(n), dtype=np.int)
    #     def get_l(ind):
    #         tmp = []
    #         for i in range(len(ind)):
    #             tmp.append([x[i][ind[i]], x[i][ind[i] + 1]])
    #         return tmp
    #     def intersection(e1, e2):
    #         start = np.max(np.vstack([e1[:, 0], e2[:, 0]]), axis=0)
    #         end = np.min(np.vstack([e1[:, 1], e2[:, 1]]), axis=0)
    #
    #         tmp = start - end
    #         args2 = np.where(tmp == 0)
    #         args1 = np.where(tmp < 0)
    #         inter = np.zeros(len(n))
    #         inter[args1] = 1
    #         inter[args2] = 2
    #         return inter
    #
    #     ns = np.product(n - 1)
    #     s = 0; i = 0
    #     while i < ns:
    #         if s == 0:
    #             for k in range(n[0] - 1):
    #                 ind[0] = k
    #                 a = get_l(ind)
    #                 l.append([i, np.array(a), p])
    #                 i += 1
    #
    #             s += 1
    #         else:
    #             while(ind[s] >= n[s] - 2):
    #                 s += 1
    #             ind[s] += 1
    #             ind[:s] = 0
    #             s = 0
    #     l = np.array(l, dtype=object)
    #     self.l = l
    #     self.neigh = neigh
    #     for i in range(len(self.l)):
    #         tmp = []
    #         for j in range(len(self.l)):
    #             if i != j:
    #                 intr = intersection(self.l[i][1], self.l[j][1])
    #                 if np.size(np.where(intr == 0)) == 0 and\
    #                             np.size(np.where(intr == 2)) == 1:
    #                     if (np.size(np.where(intr == 1)) > 0 and self.n > 1) or self.n == 1:
    #                         tmp.append([self.l[i][0], self.l[j][0]])
    #         if len(tmp) > 0:
    #             self.neigh.append(tmp)
    #     self.l = np.array(self.l)
    def generateUniformMeshOnRectange(self, rectangle, divisions, polynomialOrder):
        """Generate uniform rectangular mesh on finite rectangle.
        Rectangles are closed and are represented as
         lists of 4-tuples(lower bound, upper bound, polynomial order, mapping type = 0) for every dimension.

        Arguments:
        rectangle: array_like, shape=(N, 2) -- Array of intervals that constitute a general rectangle.
        division: array_like, shape=(N,) -- amount of divisions for every interval in general rectangle (for every dimension)
        polynomialOrder array_like, shape=(N,) -- polynomial/approximation order for every dimension

        Returns:
        Nothing, creates self.elements array_like object with shape=(product(divisions), dimension, 4) and sets the size
        of elements "list"
        """

        divisions = np.atleast_1d(divisions)
        polynomialOrder = np.atleast_1d(polynomialOrder)
        rectangle = np.atleast_2d(rectangle)
        numberOfDimensions = rectangle.shape[0]
        listOfElementBoundariesOrders = []
        for i in range(numberOfDimensions):
            leftGeneralIntervalLimit = rectangle[i, 0]
            rightGeneralIntervalLimit = rectangle[i, 1]
            numberOfDivisions = divisions[i]

            #we have numberOfDivisions + 1 because of how linspace works
            #duplicate all values after making linspace
            elementBoundaries = np.repeat(np.linspace(leftGeneralIntervalLimit, rightGeneralIntervalLimit, numberOfDivisions + 1), 2)
            #drop redundant values at boundary -> reshape arrays into (a, b) intervals
            elementBoundaries = np.reshape((elementBoundaries[1: -1]), [int((elementBoundaries.size - 2)/2), 2])
            #now we add information about order of approximating space in i's dimension
            polynomialOrders = polynomialOrder[i]*np.ones(divisions[i])
            mappingType = np.zeros(divisions[i])
            elementBoundariesOrders = np.hstack([elementBoundaries, polynomialOrders[:, None], mappingType[:, None]])
            listOfElementBoundariesOrders.append(elementBoundariesOrders)

        #take outer product of lists and we're done
        elementsList = np.array(list(itertools.product(*listOfElementBoundariesOrders)))
        self.elements = elementsList
        self.setElementsAmount()


    def setElementsAmount(self):
        self.elementsAmount = np.shape(self.elements)[0]

    def checkIntersectionOfIntervals(self, intervals, query):
        """Find intersections between intervals.
        Intervals are closed and are represented as pairs (lower bound,
        upper bound).

        Arguments:
        intervals: array_like, shape=(N, 2) -- Array of intervals.
        query: array_like, shape=(2,) -- Interval to query.

        Returns:
        Array of indexes of intervals that overlap with query.
        """
        lower, upper = query
        return np.argwhere(((lower <= intervals[:, 1]) & (intervals[:, 0] <= upper)))
    def establishNeighbours(self):
        """Find neighbours of elements, i.e. elements, such that at least one point is shared between two of them

        Arguments:
        None

        Returns:
        Nothing. Creates self.neighbours object that is a list where for every element there is a list of its neighbours
        """
        #given self.elementList calculates neighbourhoods of elements
        elementsListShape = self.elements.shape
        elementsAmount = elementsListShape[0]
        dimension = elementsListShape[1]
        elementsIntersections = []
        for i in range(elementsAmount):
            intersectionsPerDimension = []
            for d in range(dimension):
                intersections = np.array(self.checkIntersectionOfIntervals(self.elements[:, d, : -2], self.elements[i, d, :-2]))
                intersectionsPerDimension.append(intersections[intersections != i])
            intersectionOfAllDimensions = reduce(np.intersect1d, intersectionsPerDimension)
            elementsIntersections.append(intersectionOfAllDimensions)
        self.neighbours = elementsIntersections
    def getElementsAmount(self):
        """
        Arguments:
        None

        Returns:
        Amount of element domains
        """
        return self.elementsAmount

    # def file_write(self, lname, nname):
    #     f = open(lname, "w+")
    #
    #     for it in self.l:
    #         it = np.hstack(([it[0], it[1].flatten(), np.array(it[2])]))
    #         f.writelines(list(map(lambda x: str(x) + ' ', it)))
    #         f.write('\n')
    #     f.close()
    #     f = open(nname, "w+")
    #     for it in self.neigh:
    #         f.writelines(list(map(lambda x: str(x) + ' ', it)))
    #         f.write('\n')
    #     f.close()
    #     return
    # def file_read(self, lname, nname):
    #     self.l = np.genfromtxt(lname)
    #     tmp = []
    #     for it in self.l:
    #         tmp.append([it[0], np.reshape(it[1:2*self.n + 1], [self.n, 2]), np.array(it[2*self.n + 1:], dtype=np.int)])
    #     self.l = tmp
    #     f = open(nname, "r+")
    #     self.neigh = f.readlines()
    #     for i in range(len(self.neigh)):
    #         self.neigh[i] = list(filter(None, re.split(r'\[|\]| |,', self.neigh[i])))[:-1]
    #         self.neigh[i] = np.array(self.neigh[i], dtype=np.int)
    #         self.neigh[i] = np.reshape(self.neigh[i], [int(self.neigh[i].size/2), 2])
    #     f.close()
    #     return
    def fileWrite(self, elementsFileName, neighboursFileName):
        """Creates two files: one with the elements info in the form a_1, b_1, p_1, ..., a_n, b_n, p_n,
         where a_i, b_i, p_i are left and right ends of an interval, p_i is approximation order for i's component.
         Second file contains rows of neighbouring elements numbers

        Arguments:
        elementsFileName:
        neighboursFileName:

        Returns:
        Nothing
        """
        toElementsFile = open(elementsFileName, "w+")
        for element in self.elements:
            for axisInfo in element:
                toElementsFile.writelines(list(map(lambda x: str(x) + ' ', axisInfo)))
            toElementsFile.write('\n')
        toElementsFile.close()
        f = open(neighboursFileName, "w+")
        for it in self.neighbours:
            f.writelines(list(map(lambda x: str(x) + ' ', it)))
            f.write('\n')
        f.close()
    def fileRead(self, elementsFileName, neighboursFileName):
        """
        Arguments:
        elementsFileName:
        neighboursFileName:

        Returns:
        None
        Creates self.elements, self.neighbours and sets self.elementsAmount
        """
        elements = np.genfromtxt(elementsFileName)
        amountOfElements, elementRowLength = elements.shape
        self.elements = np.reshape(elements, [amountOfElements, int(elementRowLength/4), 4])
        fromNeighboursFile = open(neighboursFileName, "r+")
        neigbours = fromNeighboursFile.readlines()
        for i in range(len(neigbours)):
            neigbours[i] = list(filter(None, re.split(r'\[|\]| |,', neigbours[i])))[:-1]
            neigbours[i] = np.array(neigbours[i], dtype=int)
        self.neighbours = neigbours
        self.setElementsAmount()
        fromNeighboursFile.close()
        return
    def extendBox(self, s, i, p):
        l1 = np.array(self.box, dtype=np.float)
        if s == 1:
            l1[i, 0] = l1[i, 1]
            l1[i, 1] = np.inf


        if s == -1:
            l1[i, 1] = l1[i, 0]
            l1[i, 0] = -np.inf

        l2 = [self.l[-1, 0] + 1, l1, p]
        n = self.l[-1, 0] + 1
        self.l = np.vstack([self.l, np.array(l2, dtype=object)])
        tmp = []
        # print(self.l)
        for j in range(self.l.shape[0]):
            if self.l.shape[0] == 2:
                self.neigh.append([[j, n]])
                self.neigh.append([[n, j]])
                break
            if ((self.l[j, 1][i, 1] == l1[i, 0]) or (self.l[j, 1][i, 0] == l1[i, 1])):
                tmp.append([n, j])
                self.neigh[j].append([j, n])
        self.neigh.append(tmp)

        return
    def extendBox2d(self, i, p):
        self.extendBox(-1, i[0], p)
        self.extendBox(1, i[0], p)
        self.extendBox(-1, i[1], p)
        self.extendBox(1, i[1], p)

        l = self.l[-4:]
        tmp = []
        n = self.l[-1][0]
        tmp.append([n + 1, np.vstack([l[0][1][0], l[2][1][1]]), p])
        tmp.append([n + 2, np.vstack([l[0][1][0], l[3][1][1]]), p])
        tmp.append([n + 3, np.vstack([l[1][1][0], l[3][1][1]]), p])
        tmp.append([n + 4, np.vstack([l[1][1][0], l[2][1][1]]), p])
        for it in tmp:
            self.l = np.vstack([self.l, np.array(it)])

        def intersection(e1, e2):
            start = np.max(np.vstack([e1[:, 0], e2[:, 0]]), axis=0)
            end = np.min(np.vstack([e1[:, 1], e2[:, 1]]), axis=0)

            tmp = start - end
            args2 = np.where(tmp == 0)
            args1 = np.where(tmp < 0)
            inter = np.zeros(self.n)
            inter[args1] = 1
            inter[args2] = 2
            return inter
        self.neigh = []
        for i in range(len(self.l)):
            tmp = []
            for j in range(len(self.l)):
                if i != j:
                    intr = intersection(self.l[i][1], self.l[j][1])
                    if np.size(np.where(intr == 0)) == 0 and\
                            np.size(np.where(intr == 1)) > 0 and\
                                np.size(np.where(intr == 2)) == 1:
                        tmp.append([self.l[i][0], self.l[j][0]])
            if len(tmp) > 0:
                self.neigh.append(tmp)

        return 0
    def extendBox2dHalfSpace(self, i, p):
        self.extendBox(1, i[0], p)
        self.extendBox(1, i[1], p)

        l = self.l[-2:]
        tmp = []
        n = self.l[-1][0]
        tmp.append([n + 1, np.vstack([l[0][1][0], l[1][1][1]]), p])
        for it in tmp:
            self.l = np.vstack([self.l, np.array(it)])

        def intersection(e1, e2):
            start = np.max(np.vstack([e1[:, 0], e2[:, 0]]), axis=0)
            end = np.min(np.vstack([e1[:, 1], e2[:, 1]]), axis=0)

            tmp = start - end
            args2 = np.where(tmp == 0)
            args1 = np.where(tmp < 0)
            inter = np.zeros(self.n)
            inter[args1] = 1
            inter[args2] = 2
            return inter
        self.neigh = []
        for i in range(len(self.l)):
            tmp = []
            for j in range(len(self.l)):
                if i != j:
                    intr = intersection(self.l[i][1], self.l[j][1])
                    if np.size(np.where(intr == 0)) == 0 and\
                            np.size(np.where(intr == 1)) > 0 and\
                                np.size(np.where(intr == 2)) == 1:
                        tmp.append([self.l[i][0], self.l[j][0]])
            if len(tmp) > 0:
                self.neigh.append(tmp)

        return 0
    def customExtendBox(self, i, p):
        self.extendBox(1, i[0], p)
        self.extendBox(1, i[1], p)
        self.extendBox(-1, i[1], p)

        l = self.l[-3:]
        tmp = []
        n = self.l[-1][0]
        # print(l)
        tmp.append([n + 1, np.vstack([l[0][1][0], l[1][1][1]]), p])
        tmp.append([n + 2, np.vstack([l[0][1][0], l[2][1][1]]), p])
        for it in tmp:
            self.l = np.vstack([self.l, np.array(it)])

        def intersection(e1, e2):
            start = np.max(np.vstack([e1[:, 0], e2[:, 0]]), axis=0)
            end = np.min(np.vstack([e1[:, 1], e2[:, 1]]), axis=0)

            tmp = start - end
            args2 = np.where(tmp == 0)
            args1 = np.where(tmp < 0)
            inter = np.zeros(self.n)
            inter[args1] = 1
            inter[args2] = 2
            return inter
        self.neigh = []
        for i in range(len(self.l)):
            tmp = []
            for j in range(len(self.l)):
                if i != j:
                    intr = intersection(self.l[i][1], self.l[j][1])
                    if np.size(np.where(intr == 0)) == 0 and\
                            np.size(np.where(intr == 1)) > 0 and\
                                np.size(np.where(intr == 2)) == 1:
                        tmp.append([self.l[i][0], self.l[j][0]])
            if len(tmp) > 0:
                self.neigh.append(tmp)

        return 0
    def extendBox3d(self, i, p):
        self.extendBox(-1, i[0], p)
        self.extendBox(1, i[0], p)
        self.extendBox(-1, i[1], p)
        self.extendBox(1, i[1], p)
        self.extendBox(-1, i[2], p)
        self.extendBox(1, i[2], p)

        l = self.l[-6:]
        tmp = []
        n = self.l[-1][0]
        tmp.append([n + 1, np.vstack([l[0][1][0], l[2][1][1], l[4][1][2]]), p])
        tmp.append([n + 2, np.vstack([l[1][1][0], l[2][1][1], l[4][1][2]]), p])
        tmp.append([n + 3, np.vstack([l[0][1][0], l[3][1][1], l[4][1][2]]), p])
        tmp.append([n + 4, np.vstack([l[0][1][0], l[2][1][1], l[5][1][2]]), p])

        tmp.append([n + 5, np.vstack([l[0][1][0], l[3][1][1], l[5][1][2]]), p])
        tmp.append([n + 6, np.vstack([l[1][1][0], l[2][1][1], l[5][1][2]]), p])
        tmp.append([n + 7, np.vstack([l[1][1][0], l[3][1][1], l[4][1][2]]), p])
        tmp.append([n + 8, np.vstack([l[1][1][0], l[3][1][1], l[5][1][2]]), p])

        tmp.append([n + 9, np.vstack([l[0][1][0], l[2][1][1], self.box[2]]), p])
        tmp.append([n + 10, np.vstack([l[1][1][0], l[2][1][1], self.box[2]]), p])
        tmp.append([n + 11, np.vstack([l[0][1][0], l[3][1][1], self.box[2]]), p])
        tmp.append([n + 12, np.vstack([l[1][1][0], l[3][1][1], self.box[2]]), p])

        tmp.append([n + 13, np.vstack([l[0][1][0], self.box[1], l[4][1][2]]), p])
        tmp.append([n + 14, np.vstack([l[1][1][0], self.box[1], l[4][1][2]]), p])
        tmp.append([n + 15, np.vstack([l[0][1][0], self.box[1], l[5][1][2]]), p])
        tmp.append([n + 16, np.vstack([l[1][1][0], self.box[1], l[5][1][2]]), p])

        tmp.append([n + 17, np.vstack([self.box[0], l[2][1][1], l[4][1][2]]), p])
        tmp.append([n + 18, np.vstack([self.box[0], l[3][1][1], l[4][1][2]]), p])
        tmp.append([n + 19, np.vstack([self.box[0], l[2][1][1], l[5][1][2]]), p])
        tmp.append([n + 20, np.vstack([self.box[0], l[3][1][1], l[5][1][2]]), p])
        for it in tmp:
            self.l = np.vstack([self.l, np.array(it)])
        def intersection(e1, e2):
            start = np.max(np.vstack([e1[:, 0], e2[:, 0]]), axis=0)
            end = np.min(np.vstack([e1[:, 1], e2[:, 1]]), axis=0)

            tmp = start - end
            args2 = np.where(tmp == 0)
            args1 = np.where(tmp < 0)
            inter = np.zeros(self.n)
            inter[args1] = 1
            inter[args2] = 2
            return inter
        self.neigh = []
        for i in range(len(self.l)):
            tmp = []
            for j in range(len(self.l)):
                if i != j:
                    intr = intersection(self.l[i][1], self.l[j][1])
                    if np.size(np.where(intr == 0)) == 0 and\
                            np.size(np.where(intr == 1)) > 0 and\
                                np.size(np.where(intr == 2)) == 1:
                        tmp.append([self.l[i][0], self.l[j][0]])
            if len(tmp) > 0:
                self.neigh.append(tmp)

        return 0
    def __getitem__(self, item):
        return [self.l[item][1], self.l[item][2]]
    def intersection(self, e1, e2):
            start = np.max(np.vstack([e1[:, 0], e2[:, 0]]), axis=0)
            end = np.min(np.vstack([e1[:, 1], e2[:, 1]]), axis=0)
            res = (np.vstack([start, end]).T)
            return res

    ##returns function sigma(a_i) = (h_i + h_{i+1})/(p_i^2 + p_{i+1}^2)
    def sigma1d(self, C=1):
        sigmaf = []
        a = []
        for i in range(len(self.l) - 1):
            a.append(self.l[i][1][0][1])
            if self.l[i + 1][1][0][1] != np.inf:

                hl = self.l[i][1][0][1] - self.l[i][1][0][0]
                hr = self.l[i + 1][1][0][1] - self.l[i + 1][1][0][0]
            else:
                hl = self.l[i][1][0][0]; hr = 1

            pl = self.l[i][2][0]
            pr = self.l[i + 1][2][0]

            sigmafn = hl + hr
            sigmafd = pl**2 + pr**2
            sigmaf.append(sigmafd/sigmafn*C)

        def sigma(x):
            res = np.zeros(x.shape)
            for i in range(len(a)):
                res[np.where(x == a[i])] = sigmaf[i]
            return res
        return sigma
    ####returns constants Ct, Cg s.t. Ct*d(K_i) <= d(a_i) <= Cg*d(K_i)
    def CaCs(self):
        a = []
        Ct = []; Cg = []
        for i in range(len(self.l) - 1):
            a.append(self.l[i][1][0][1])
            if self.l[i + 1][1][0][1] != np.inf:

                hl = self.l[i][1][0][1] - self.l[i][1][0][0]
                hr = self.l[i + 1][1][0][1] - self.l[i + 1][1][0][0]
            else:
                hl = self.l[i][1][0][0]; hr = 1

            pl = self.l[i][2][0]
            pr = self.l[i + 1][2][0]

            fn = hl + hr;       fnl = hl;       fnr = hr
            fd = pl**2 + pr**2; fdl = pl**2 ;   fdr = pr**2

            left = fnl/fdl; mid = 1/2*fn/fd; right = fnr/fdr
            Ct.append(mid/left); Cg.append(mid/right)

        Ct = np.array(Ct, dtype=np.float); Cg = np.array(Cg, dtype=np.float)
        # print(Cg)
            # print('now ', i)
            # print(left, mid, right)
            # print(ct, cg)
            # print(left*ct, mid, right*cg)
        Ct = np.min(Ct); Cg = np.max(Cg)
        # print(np.min(Ct), np.max(Cg))
        Caab = 2*np.max([3 + 4*np.sqrt(5/3), 2 + 2*np.sqrt(10/3)])
        Cs = Cg*4*Caab
        C1 = 1 + Cs*Ct + 7*Cg/Cs
        C2 = 7*Cs*Ct
        Cr = np.max([C1, C2])
        Cf = np.max([2, Cg/np.sqrt(3)/Cs])
        Ca = 4*(np.sqrt(2) + 1)*np.sqrt(Cr)*Cf
        return Ca, Cs
        # print(C1, C2)
        # print(self.l)
    def sigma_list(self, Cs):
        a = []
        sigma_l = []
        for i in range(len(self.l) - 1):
            a.append(self.l[i][1][0][1])
            if self.l[i + 1][1][0][1] != np.inf:

                hl = self.l[i][1][0][1] - self.l[i][1][0][0]
                hr = self.l[i + 1][1][0][1] - self.l[i + 1][1][0][0]
            else:
                hl = self.l[i][1][0][0]; hr = 1

            pl = self.l[i][2][0]
            pr = self.l[i + 1][2][0]

            fn = hl + hr;       fnl = hl;       fnr = hr
            fd = pl**2 + pr**2; fdl = pl**2 ;   fdr = pr**2
            sigma_l.append(Cs/(1/2*fn/fd))
        return sigma_l

# meshObj = mesh(2)
# meshObj.generateUniformMeshOnRectange(rectangle=[[0, 1], [0, 1]], divisions=[3, 3], polynomialOrder=[5, 6])
# meshObj.establishNeighbours()
# meshObj.fileWrite("elements.txt", "neighbours.txt")
# meshObj.fileRead("elements.txt", "neighbours.txt")

