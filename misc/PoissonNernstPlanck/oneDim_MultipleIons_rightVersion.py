import numpy as np
from mathematics import spectral as spec
from mathematics import integrate as integr
import scipy.linalg as sp
import matplotlib.pyplot as plt
import time as time
import warnings
warnings.filterwarnings("ignore")
n = 50
ionsAmount = 3
z = np.array([1.0, 1.0, -1.0])
# concentration_coeffs = np.array([1.0, 0.04, 0.45])
diffusion_coeffs = np.array([1.96, 1.33, 2.03])
diffusion_coeffs = np.ones(ionsAmount, dtype=float)
D = spec.chebDiffMatrix(n, 0, 1)
I = np.eye(n)
x = spec.chebNodes(n, 0, 1)
R = 8.31431
T = 18.0 + 273.0
f = 9.64853
F = f * 10 ** 4
l = 5.0
e = 8.854187
e_r = 2
alpha = f * f * l * l / (R * T * e * e_r) * 100
partition_coefficients = np.array([1.0, 0.04, 0.45])#/400.0
# V_r = 90.0
# C0 = np.array([3.0, 152.0, 180.0])#* partition_coefficients
# # C1 = np.array([345.0, 72.0, 61.0]) * partition_coefficients\
# C1 = np.array([345.0, 72.0, 61.0])# * partition_coefficients

C0 = np.array([20.0, 445.0, 587.0])
C1 = np.array([345.0, 72.0, 61.0])
phi0 = 0.0

weights, nodes = integr.reg_22_wn(0, 1, 1000)
np.set_printoptions(precision=3, suppress=True)
# V_r = 29.8
V_r = 3.7
phi1 = F * V_r / 1000 / R / T

potential = x.copy() * phi1
concentrations = C1[:, np.newaxis] * x.copy()[np.newaxis, :] + C0[:, np.newaxis] * (1 - x.copy()[np.newaxis, :])

vectorPhi = potential
phiD = D @ vectorPhi
phiD2 = D @ phiD
phiD_Function = lambda x: spec.barycentricChebInterpolate(phiD, x, 0, 1, extrapolation=1)
evaluatedConcentrations = spec.barycentricChebInterpolate(concentrations.T, nodes, 0, 1, 1, axis=0)
fluxes = diffusion_coeffs*(-np.sum(phiD_Function(nodes) * evaluatedConcentrations.T * weights, axis=1)
                           - z*(concentrations[:, -1] - concentrations[:, 0]))

concentrationOperators = [None] * ionsAmount
for currentConcentrationIndex in range(ionsAmount):
    concentrationOperators[currentConcentrationIndex] = D


del currentConcentrationIndex
concentrationOperators = np.array(concentrationOperators)
fluxes = np.array(fluxes)
print("initial fluxes:", fluxes)
potentialOperator = D @ D
potentialOperator[0, :] = 0.0
potentialOperator[-1, :] = 0.0

potentialOperator[0, 0] = 1.0
potentialOperator[-1, -1] = 1.0

for globalIterationIndex in range(100):
    RHS = np.zeros([ionsAmount, n], dtype=float)
    """CII := Concentration Iteration Index"""
    for CII in range(ionsAmount):
        concentrationOperators[CII] =\
            D @ D + z[CII] * (np.diag(phiD2) + np.diag(phiD) @ D)
        concentrationOperators[CII, 0, :] = 0.0
        concentrationOperators[CII, 0, 0] = 1.0

        concentrationOperators[CII, -1, :] = 0.0
        concentrationOperators[CII, -1, -1] = 1.0

        rhs = np.zeros(n, dtype=float)
        RHS[CII, 0] = C0[CII]
        RHS[CII, -1] = C1[CII]

    fullMatrix = sp.block_diag(*concentrationOperators)
    fullRHS = RHS.flatten()
    concentrations = sp.solve(fullMatrix, fullRHS)
    concentrations = np.reshape(concentrations, [ionsAmount, n])

    evaluatedConcentrations = spec.barycentricChebInterpolate(concentrations.T, nodes, 0, 1, 1, axis=0)
    fluxes = - (z * phiD[0] * C0 + (D @ concentrations.T)[-1, :])
    print(fluxes, np.sum(fluxes * partition_coefficients))
    potentialRHS = -(alpha) * np.sum(z * concentrations.T, axis=1) * 0
    potentialRHS[0] = phi0
    potentialRHS[-1] = phi1
    potential = sp.solve(potentialOperator, potentialRHS)
    phiD = D @ potential
    phiD2 = D @ phiD

print("fluxes: ", fluxes)
plt.plot(nodes, evaluatedConcentrations)
plt.show()

# print(concentration)

# time.sleep(500)
# for C0 in [3.0]:
#     concentration_values = []
#     for V_r in spec.chebNodes(20, 0, 120):
#
#         for i in range(20):
#             vectorPhi = potential
#             phiD = D @ vectorPhi
#             I = np.eye(n)
#             concentrationOperator = -(D) - (np.diag(phiD))
#             concentrationOperator[0, :] = 0
#             concentrationOperator[0, 0] = 1.0
#             # concentrationOperator[-1, -1] = 1.0
#             # print(concentrationOperator)
#             concentrationRHS = -np.ones(n, dtype=float) * 1
#             concentrationRHS[0] = C0
#             # print(concentrationRHS)
#             # concentrationRHS[-1] = C1
#             # print(concentrationOperator)
#             concentration = sp.solve(concentrationOperator, concentrationRHS)
#
#             # plt.plot(x, concentration)
#             # plt.show()
#             # time.sleep(500)
#             potentialOperator = D @ D
#             potentialOperator[0, :] = 0.0
#             # potentialOperator[-1, :] = D[-1, :]
#             potentialOperator[-1, :] = 0.0
#
#             potentialOperator[0, 0] = 1.0
#             potentialOperator[-1, -1] = 1.0
#
#             potentialRHS = -(alpha) * z * concentration
#             potentialRHS[0] = phi0
#             potentialRHS[-1] = phi1
#             prevPotential = potential.copy()
#             # potential = 0.5*(prevPotential + sp.solve(potentialOperator, potentialRHS))
#             potential = (sp.solve(potentialOperator, potentialRHS))
#             # plt.plot(x, potential)
#             # plt.show()
#             error = np.max(prevPotential - potential)
#             # print(V_r, error)
#
#     concentration_values = np.array(concentration_values)
#     plt.plot(concentration_values[:, 0], concentration_values[:, 1], color='red', label='Concentration')
#     plt.plot(concentration_values[:, 0], concentration_values[:, 2], color='blue',  label='Flux')
#     plt.plot(concentration_values[:, 0], concentration_values[:, 3], color='orange',  label='Nernst concentration')
# plt.legend(fontsize=14)
# plt.grid(True)
# plt.show()