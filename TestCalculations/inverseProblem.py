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

integrationPointsAmount = 2000
approximationOrder = 100
galerkinMethodObject = galerkin.GalerkinMethod1d(methodType=galerkin.GalerkinMethod1d.MethodType.SpectralLinearSystem)

k = lambda x: 1.0
psi = lambda x: np.sin(2*np.pi*x)
eta = lambda t: np.exp(-5*t)

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
h = 0.01

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
# plt.plot(points[1:-1], solutions[int(int(0.1/h)/100), :])
# plt.plot(points[1:-1], solutions[int(int(0.1/h)/3), :])
# plt.plot(points[1:-1], solutions[int(int(0.1/h)/2), :])
plt.plot(points[1:-1], solutions[0, :])
plt.plot(points[1:-1], solutions[int(1.0/h / 2), :])
plt.plot(points[1: -1], solutions[-1, :])
plt.show()
plt.plot(solutions.T[-1, :])
plt.plot(solutions.T[0, :])
plt.show()
plt.imshow(solutions)
plt.show()
# for it in ms:
#     print(it)
# for it in fs:
#     print(it)