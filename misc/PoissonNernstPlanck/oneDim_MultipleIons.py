import numpy as np
from mathematics import spectral as spec
from mathematics import integrate as integr
import scipy.linalg as sp
import matplotlib.pyplot as plt
import time as time
import warnings
warnings.filterwarnings("ignore")
n = 5
ionsAmount = 3
z = [1.0, 1.0, -1.0]
concentration_coeffs = [1.0, 0.04, 0.45]
D = spec.chebDiffMatrix(n, 0, 1)
I = np.eye(n)
x = spec.chebNodes(n, 0, 1)
R = 8.31431
T = 24.0 + 273.0
f = 9.64853
F = f * 10 ** 4
l = 9.0
e = 8.854187
e_r = 20
alpha = f * f * l * l / (R * T * e * e_r) * 100
potential_values = spec.chebNodes(50, 1, 150)

# V_r = 90.0
C0 = np.array([3.0, 152.0, 180.0])
C1 = np.array([345.0, 72.0, 61.0])
phi0 = 0.0;
# phi1 = F * V_r / 1000 / R / T

# print(phi1, alpha)
w, nodes = integr.reg_22_wn(0, 1, 1000)
np.set_printoptions(precision=3, suppress=True)
V_r = 60
phi1 = F * V_r / 1000 / R / T

potential = x.copy() * phi1
concentrations = C1[:, np.newaxis] * x.copy()[np.newaxis, :] + C0[:, np.newaxis] * (1 - x.copy()[np.newaxis, :])

vectorPhi = potential
phiD = D @ vectorPhi
concentrationOperators = [None] * ionsAmount
fluxes = [None] * ionsAmount
for currentConcentrationIndex in range(ionsAmount):
    concentrationOperators[currentConcentrationIndex] = ((-concentration_coeffs[currentConcentrationIndex]*
                             ((D) + z[currentConcentrationIndex]*(np.diag(phiD)))))
    fluxes[currentConcentrationIndex] = concentrationOperators[currentConcentrationIndex] @ concentrations[currentConcentrationIndex]
del currentConcentrationIndex
concentrationOperators = np.array(concentrationOperators)
fluxes = np.array(fluxes)
potentialOperator = D @ D
potentialOperator[0, :] = 0.0
potentialOperator[-1, :] = 0.0

potentialOperator[0, 0] = 1.0
potentialOperator[-1, -1] = 1.0
for globalIterationIndex in range(100):
    for concentrationInterationIndex in range(ionsAmount):

        concentrationOperators[concentrationInterationIndex] = ((-concentration_coeffs[concentrationInterationIndex]*
                                 ((D) + z[concentrationInterationIndex]*(np.diag(phiD)))))
        # print(concentrationOperators[0])
        concentrationOperators[concentrationInterationIndex, 0, :] = 0.0
        concentrationOperators[concentrationInterationIndex, 0, 0] = 1.0
        fluxesSum = np.sum(fluxes[:concentrationInterationIndex, :], axis=0) + np.sum(fluxes[concentrationInterationIndex + 1:, :], axis=0)
        fluxesSum[0] = -C0[concentrationInterationIndex]
        concentrations[concentrationInterationIndex] = sp.solve(concentrationOperators[concentrationInterationIndex], -fluxesSum)
        print(concentrationInterationIndex)
        print(concentrations[concentrationInterationIndex])
        plt.plot(x, concentrations[concentrationInterationIndex, :])
        plt.show()
        fluxCalculationOperator = ((-concentration_coeffs[concentrationInterationIndex] *
          ((D) + z[concentrationInterationIndex] * (np.diag(phiD)))))

        fluxes[concentrationInterationIndex] = fluxCalculationOperator @ concentrations[concentrationInterationIndex]
        # print(concentrations[concentrationInterationIndex])
        # print(fluxes[concentrationInterationIndex])
        print(fluxes)
        potentialRHS = -(alpha) * np.sum(z * concentrations.T, axis=1)
        potentialRHS[0] = phi0
        potentialRHS[-1] = phi1
        potential = sp.solve(potentialOperator, potentialRHS)
        vectorPhi = potential
        phiD = D @ vectorPhi
        # if(globalIterationIndex % 3 == 0):

        plt.plot(x, potential)
        plt.show()



        # time.sleep(500)

# print(concentration)

time.sleep(500)
for C0 in [3.0]:
    concentration_values = []
    for V_r in spec.chebNodes(20, 0, 120):

        for i in range(20):
            vectorPhi = potential
            phiD = D @ vectorPhi
            I = np.eye(n)
            concentrationOperator = -(D) - (np.diag(phiD))
            concentrationOperator[0, :] = 0
            concentrationOperator[0, 0] = 1.0
            # concentrationOperator[-1, -1] = 1.0
            # print(concentrationOperator)
            concentrationRHS = -np.ones(n, dtype=float) * 1
            concentrationRHS[0] = C0
            # print(concentrationRHS)
            # concentrationRHS[-1] = C1
            # print(concentrationOperator)
            concentration = sp.solve(concentrationOperator, concentrationRHS)

            # plt.plot(x, concentration)
            # plt.show()
            # time.sleep(500)
            potentialOperator = D @ D
            potentialOperator[0, :] = 0.0
            # potentialOperator[-1, :] = D[-1, :]
            potentialOperator[-1, :] = 0.0

            potentialOperator[0, 0] = 1.0
            potentialOperator[-1, -1] = 1.0

            potentialRHS = -(alpha) * z * concentration
            potentialRHS[0] = phi0
            potentialRHS[-1] = phi1
            prevPotential = potential.copy()
            # potential = 0.5*(prevPotential + sp.solve(potentialOperator, potentialRHS))
            potential = (sp.solve(potentialOperator, potentialRHS))
            # plt.plot(x, potential)
            # plt.show()
            error = np.max(prevPotential - potential)
            # print(V_r, error)

    concentration_values = np.array(concentration_values)
    plt.plot(concentration_values[:, 0], concentration_values[:, 1], color='red', label='Concentration')
    plt.plot(concentration_values[:, 0], concentration_values[:, 2], color='blue',  label='Flux')
    plt.plot(concentration_values[:, 0], concentration_values[:, 3], color='orange',  label='Nernst concentration')
plt.legend(fontsize=14)
plt.grid(True)
plt.show()