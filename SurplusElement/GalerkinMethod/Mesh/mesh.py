import itertools as itertools
from functools import reduce
# from SurplusElement.GalerkinMethod.element.Element1d.element1d import ElementType
import numpy as np
import re

class mesh():
    def __init__(self, n):
        self.n = n
        return

    def generateUniformMeshOnRectange(self, rectangle, divisions, polynomialOrder):
        """Generate uniform rectangular Mesh on finite rectangle.
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

            #numberOfDivisions + 1 because of how linspace works
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

    def generateProvidedMeshOnRectange(self, rectangleMesh, polynomialOrder, mappingType):
        """
        """

        polynomialOrder = np.atleast_2d(polynomialOrder)
        rectangle = np.atleast_2d(rectangleMesh)
        mappingType = np.atleast_2d(mappingType)
        numberOfDimensions = rectangle.shape[0]
        listOfElementBoundariesOrders = []
        for i in range(numberOfDimensions):

            elementBoundaries = np.repeat(rectangle[i, :], 2)
            #drop redundant values at boundary -> reshape arrays into (a, b) intervals
            elementBoundaries = np.reshape((elementBoundaries[1: -1]), [int((elementBoundaries.size - 2)/2), 2])
            #now we add information about order of approximating space in i's dimension
            elementBoundariesOrders = np.hstack([elementBoundaries, polynomialOrder[i, :, np.newaxis], mappingType[i, :, np.newaxis]])
            listOfElementBoundariesOrders.append(elementBoundariesOrders)

        #take outer product of lists and we're done
        elementsList = np.array(list(itertools.product(*listOfElementBoundariesOrders)))
        self.elements = elementsList
        self.setElementsAmount()
    def generateSkewedMeshOnRectange(self, rectangle, divisions, polynomialOrder):
        """
        NO DESCRIPTION
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

            #numberOfDivisions + 1 because of how linspace works
            #duplicate all values after making linspace
            x = np.linspace(0, 1, numberOfDivisions + 1)
            x = leftGeneralIntervalLimit + (rightGeneralIntervalLimit - leftGeneralIntervalLimit) * x ** 4
            elementBoundaries = np.repeat(x, 2)
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

    def fileWrite(self, elementsFileName: str, neighboursFileName: str):
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

    def fileRead(self, elementsFileName: str, neighboursFileName: str):
        """
        Arguments:
        elementsFileName:
        neighboursFileName:

        Returns:
        None
        Creates self.elements, self.neighbours and sets self.elementsAmount
        """
        elements = np.genfromtxt(elementsFileName)
        elements = np.atleast_2d(elements)
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


    # def extendRectangleToInf_AlongAxis_OneDirection(self, direction: str, infElementBoundary: float, axis: int, approximationOrder: int):
    #     elementsAmount = self.getElementsAmount()
    #     offset = 0
    #     if direction == "right":
    #         for elementNumber in range(elementsAmount):
    #             interval = self.elements[elementNumber + offset, axis, :2]
    #             if interval[1] == infElementBoundary:
    #                 infElement = self.elements[elementNumber + offset].copy()
    #                 infElement[axis, 0] = infElementBoundary
    #                 infElement[axis, 1] = np.inf
    #                 infElement[axis, 2] = approximationOrder
    #                 infElement[axis, -1] = ElementType.RATIONAL_INF_HALF_SPACE.value
    #                 self.elements = np.insert(self.elements, elementNumber + 1 + offset, infElement, axis=0)
    #                 offset += 1


