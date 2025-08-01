import numpy as np
import SurplusElement.GalerkinMethod.Mesh.mesh as MeshClass
import SurplusElement.GalerkinMethod.Galerkin1d as galerkin
import SurplusElement.GalerkinMethod.element.Element1d.element1dUtils as elem1dUtils
import time as time
from SurplusElement.mathematics import integrate as integr
import matplotlib.pyplot as plt
import scipy.optimize as sp_opt
import SurplusElement.mathematics.spectral as spec
from scipy import integrate as integrate
import scipy.linalg as sp_linalg
from scipy.interpolate import CubicSpline
integrationPointsAmount = 2000
approximationOrder = 120
galerkinMethodObject = galerkin.GalerkinMethod1d(methodType=galerkin.GalerkinMethod1d.MethodType.SpectralLinearSystem)

k = lambda x: 1.0
psi = lambda x: np.sin(10*np.pi*x)
# psi = lambda x: x
# eta = lambda t: np.exp(-5*t)
# eta = lambda t: np.exp(-t)
eta = lambda t: 10*np.exp(5*t)

gradForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm1(
        trialElement, testElement, lambda x: k(x), integrationPointsAmount)
innerForm = lambda trialElement, testElement: elem1dUtils.integrateBilinearForm0(
        trialElement, testElement, lambda x: 1.0, integrationPointsAmount)

functional = lambda testElement: elem1dUtils.integrateFunctional(
        testElement=testElement, function=lambda x: psi(x), weight=lambda x: 1.0, integrationPointsAmount=integrationPointsAmount)

galerkinMethodObject.setBilinearForm(innerForms=[innerForm, gradForm], boundaryForms=[])
galerkinMethodObject.setRHSFunctional([functional])

boundaryConditions = ['{"boundaryPoint": "0", "boundaryValue": 0.0}',
                          '{"boundaryPoint": "1.0", "boundaryValue": 0.0}']
# boundaryConditions = []

galerkinMethodObject.setDirichletBoundaryConditions(boundaryConditions)

mesh = MeshClass.mesh(1)
f = open("elementsDataSpectral.txt", "w")
f.write("0.0 1.0 " + str(approximationOrder) + " 0.0")
f.close()
mesh.fileRead("elementsDataSpectral.txt")
galerkinMethodObject.initializeMesh(mesh)
galerkinMethodObject.initializeElements()
galerkinMethodObject.calculateElements()
ms, fs = galerkinMethodObject.getMatrices()
M = ms[0][1:-1, 1:-1]
L = ms[1][1:-1, 1:-1]
b = fs[0][1:-1]
invM = sp_linalg.inv(M)

A = invM @ L
B = invM @ b
h = 0.0001

AA = h * A + np.eye(approximationOrder - 2)
BB = h * B


x0 = np.zeros(approximationOrder - 2)
xPrev = x0
solutions = [xPrev]
for i in range(int(1.0/h)):
    xNext = sp_linalg.solve(AA, xPrev + BB * eta((i + 1) * h))
    solutions.append(xNext)
    # xNext = AA @ xPrev + BB * eta(i * h)
    xPrev = xNext
solutions = np.array(solutions)
points = galerkinMethodObject.getMeshPoints()
def plotDirectSolution():
    plt.plot(points[1:-1], solutions[int(int(0.1/h)/100), :])
    plt.plot(points[1:-1], solutions[int(int(0.1/h)/3), :])
    plt.plot(points[1:-1], solutions[int(int(0.1/h)/2), :])
    plt.plot(points[1:-1], solutions[0, :])
    plt.plot(points[1:-1], solutions[int(1.0/h / 2), :])
    plt.plot(points[1: -1], solutions[-1, :])
    plt.show()
    plt.plot(solutions.T[-1, :])
    plt.plot(solutions.T[0, :])
    plt.show()
    plt.imshow(solutions)
    plt.show()
x8 = int(approximationOrder/100.0)
x8 = 100
c = galerkinMethodObject.evaluateBasisAtPoints(points=points)
c = (c[:, 1: -1])[x8, :]
# print(c.shape)
phi = solutions[:, x8]
cs = CubicSpline((np.arange(int(1.0/h) + 1)) * h, phi)
dphi = cs.derivative(1)((np.arange(int(1.0/h) + 1)) * h)
# print((np.arange(int(1.0/h) + 1)) * h)
# plt.plot(phi)
# plt.show()
# plt.plot(dphi)
# plt.plot(np.diff(phi)/h)
# plt.show()
# dphi = np.diff(phi)/h

np.set_printoptions(precision=3, suppress=True)
cb = c @ B
x0 = np.zeros(approximationOrder - 2)
xPrev = x0
solutionsNew = [xPrev]
AA = h * A + np.eye(approximationOrder - 2) - np.outer(BB, (c @ A) / cb)
for i in range(int(1.0/h)):
    # xNext = sp_linalg.solve(AA, xPrev + BB * (c @ (A @ xPrev))/cb + BB / cb * dphi[i])
    xNext = sp_linalg.solve(AA, xPrev + BB / cb * dphi[i])
    # print(BB * (c @ (A @ xPrev))/cb, BB / cb * dphi[i])
    solutionsNew.append(xNext)
    # xNext = AA @ xPrev + BB * eta(i * h)
    xPrev = xNext
solutionsNew = np.array(solutionsNew)
def plotInverseSolution():
    plt.plot(points[1:-1], solutionsNew[int(int(0.1/h)/100), :])
    plt.plot(points[1:-1], solutionsNew[int(int(0.1/h)/3), :])
    plt.plot(points[1:-1], solutionsNew[int(int(0.1/h)/2), :])
    plt.plot(points[1:-1], solutionsNew[0, :])
    plt.plot(points[1:-1], solutionsNew[int(1.0/h / 2), :])
    plt.plot(points[1: -1], solutionsNew[-1, :])
    plt.show()
    plt.plot(solutionsNew.T[-1, :])
    plt.plot(solutionsNew.T[0, :])
    plt.show()
    plt.imshow(solutionsNew)
    plt.show()
# plotDirectSolution()
# plotInverseSolution()

plt.plot(solutionsNew[-1, :])
plt.plot(solutions[-1, :])
plt.show()
#
# plt.plot(solutionsNew[-1, :] - solutions[-1, :])
# plt.show()