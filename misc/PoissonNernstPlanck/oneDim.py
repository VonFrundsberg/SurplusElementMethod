# import numpy as np
# from mathematics import spectral as spec
# from mathematics import integrate as integr
# import scipy.linalg as sp
# import matplotlib.pyplot as plt
# import time as time
#
# n = 80
# # WORKS ONLY FOR Z = 1
# z = 1
# D = spec.chebDiffMatrix(n, 0, 1)
# I = np.eye(n)
# x = spec.chebNodes(n, 0, 1)
# R = 8.31431
# T = 24.0 + 273.0
# f = 9.64853
# F = f * 10 ** 4
# l = 9.0
# e = 8.854187
# e_r = 20
# alpha = f * f * l * l / (R * T * e * e_r) * 100
# potential_values = spec.chebNodes(50, 1, 150)
# concentration_values = []
# for V_r in potential_values:
#     # V_r = 90.0
#     C0 = 3.0;
#     C1 = 99.0;
#     phi0 = 0.0;
#     phi1 = F * V_r / 1000 / R / T
#
#
#     # print(phi1, alpha)
#     potential = x.copy()
#     concentration = x.copy() + C0
#     for i in range(50):
#         vectorPhi = potential
#         phiFunction = lambda x: spec.barycentricChebInterpolate(vectorPhi, x, 0, 1, extrapolation=1)
#         phiD = D @ vectorPhi
#         concentrationOperator = -D - z * I * phiD
#         concentrationOperator[0, :] = 0
#         concentrationOperator[-1, :] = 0
#         concentrationOperator[0, 0] = 1.0
#         concentrationOperator[-1, -1] = 1.0
#         j = -0*(C1 * np.exp(phi1) - C0 * np.exp(phi0)) / integr.reg_32(
#             lambda x: np.exp(z * phiFunction(x)), a=0, b=1, n=1000);
#         concentrationRHS = j * np.ones(n)
#         concentrationRHS[0] = C0
#         concentrationRHS[-1] = C1
#         concentration = sp.solve(concentrationOperator, concentrationRHS)
#
#         potentialOperator = D @ D
#         potentialOperator[0, :] = 0.0
#         # potentialOperator[-1, :] = D[-1, :]
#         # potentialOperator[-1, :] = 0.0
#
#         potentialOperator[0, 0] = 1.0
#         # potentialOperator[-1, -1] = 1.0
#
#         potentialRHS = -(alpha) * z * concentration
#         potentialRHS[0] = phi0
#         # potentialRHS[-1] = phi1
#         prevPotential = potential.copy()
#         # potential = 0.5*(prevPotential + sp.solve(potentialOperator, potentialRHS))
#         potential = (sp.solve(potentialOperator, potentialRHS))
#         error = np.max(np.abs(prevPotential - potential))
#
#         # plt.plot(x, concentration)
#         # plt.plot(x, potential)
#         # plt.show()
#     print(potential[-1])
#     concentration_values.append([error, j, concentration[-1]])
#     # print("calculated experimental Nernst Vm: ", R * T / F / z * np.log(concentration[0]/99.0)*1000)
#     # print("alpha")
#     # print("concentraction values are: ", concentration[0], concentration[-1])
#     # # print(potential[0], potential[-1])
#     # x_values = x
#     # plt.rcParams['font.family'] = 'Times New Roman'
#     # # Plot for potential
#     # # Plot for potential
#     # plt.figure(figsize=(10, 6))
#     # plt.plot(x_values, potential, label='Potential', color='blue')
#     # plt.title('Potential Distribution', fontsize=20)  # Increase title font size
#     # plt.xlabel('Position', fontsize=16)  # Increase x-axis label font size
#     # plt.ylabel('Potential', fontsize=16)  # Increase y-axis label font size
#     # plt.grid(True)
#     # plt.legend(fontsize=14)  # Increase legend font size
#     # plt.tight_layout()
#     # plt.show()
#     #
#     # # Plot for concentration
#     # plt.figure(figsize=(10, 6))
#     # plt.plot(x_values, concentration, label='Concentration', color='red')
#     # plt.title('Concentration Distribution', fontsize=20)  # Increase title font size
#     # plt.xlabel('Position', fontsize=16)  # Increase x-axis label font size
#     # plt.ylabel('Concentration', fontsize=16)  # Increase y-axis label font size
#     # plt.grid(True)
#     # plt.legend(fontsize=14)  # Increase legend font size
#     # plt.tight_layout()
#     # plt.show()
# # plt.plot(potential_values, concentration_values)
# #
# # plt.show()
# errors, js, concentrations = zip(*concentration_values)
#
# # Plot
# plt.figure(figsize=(10, 6))
# plt.plot(potential_values, errors, marker='o', linestyle='-', color='b', label='Errors')
# plt.plot(potential_values, js, marker='s', linestyle='--', color='r', label='js')
# plt.plot(potential_values, concentrations, marker='^', linestyle='-.', color='g', label='Concentrations')
#
# # Add labels and title
# plt.xlabel('Potential Values')
# plt.ylabel('Values')
# plt.title('Errors, js, and Concentrations vs. Potential with C0 = {}'.format(C0))
# plt.grid(True)
# plt.legend()
#
# # Show plot
# plt.tight_layout()
# plt.show()

import numpy as np
from mathematics import spectral as spec
from mathematics import integrate as integr
import scipy.linalg as sp
import matplotlib.pyplot as plt
import time as time
import warnings
warnings.filterwarnings("ignore")
from numpy.polynomial import chebyshev as cheb_poly
n = 5
# WORKS ONLY FOR Z = 1
z = 1
D = spec.chebDiffMatrix(n, 0, 1)
I = np.eye(n)
x = spec.chebNodes(n, 0, 1)
R = 8.31431
T = 24.0 + 273.0
f = 9.64853
F = f * 10 ** 4
l = 9.0
e = 8.854187
e_r = 10
alpha = f * f * l * l / (R * T * e * e_r) * 100
potential_values = spec.chebNodes(50, 1, 150)

# V_r = 90.0
C0 = 3.0;
C1 = 99.0;
phi0 = 0.0;
# phi1 = F * V_r / 1000 / R / T

# print(phi1, alpha)
potential = x.copy()
w, nodes = integr.reg_22_wn(0, 1, 1000)
np.set_printoptions(precision=3, suppress=True)
for C0 in [3.0]:
    concentration_values = []
    for V_r in spec.chebNodes(20, 0, 120):
        phi1 = F * V_r / 1000 / R / T
        print(phi1)
        concentration = C1 * x.copy() + C0 * (1 - x.copy())
        for i in range(20):
            vectorPhi = potential
            phiD = D @ vectorPhi
            phiFunction = lambda x: spec.barycentricChebInterpolate(phiD, x, 0, 1, extrapolation=1)

            I = np.eye(n)
            # concentrationOperator = -(np.diag(x) @ D) - (I @ np.diag(x) @ np.diag(phiD)) + I
            concentrationOperator = -(D) - (np.diag(phiD))
            print(concentrationOperator)
            # print()
            # integralElementF = lambda x: (spec.barycentricChebInterpolate(I, x, a=0, b=1, extrapolation=1).T *
            #                    phiFunction(x))
            # prevX = x[0]
            # integrals = []
            # for currentX in x[1:]:
            #     integrals.append(integr.reg_32(f=integralElementF, a=prevX, b=currentX, n=100))
            #     # print("integrals over: ", [prevX, currentX])
            #     # print(integr.reg_32(f=integralElementF, a=prevX, b=currentX, n=10))
            #     prevX = currentX
            # integralsArr = np.array(integrals)
            # integralsArr = np.vstack([np.zeros([1, n]), np.array(integrals)])
            # # print("before cumsum")
            # # print(integralsArr)
            # integralsArr = np.cumsum(integralsArr, axis=0)
            # concentrationOperator += integralsArr
            # print("after cumsum")
            # print(integralsArr)
            concentrationOperator[0, :] = 0
            # concentrationOperator[-1, :] = 0
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
        # print(error)
        # print(concentration[-1])
        # plt.plot(x, concentration)
        # plt.plot(x, np.exp(-6*x)*C0)
        # concentration_xs = spec.barycentricChebInterpolate(concentration, nodes, a=0, b=1, extrapolation=1)
        # concentration_xs *= (concentration_xs <= 0)
        # errorTerm = np.sum(w * concentration_xs)
        # print(errorTerm)
        # plt.plot(x, potential)
        # plt.show()
        # print(potential[-1])
        # concentration_values.append([error, j, concentration[-1]])
        # print("calculated experimental Nernst Vm: ", R * T / F / z * np.log(concentration[0] / 99.0) * 1000)
        # print("alpha")
        NernstConcentration = concentration[0]/np.exp(F/R/T*z*V_r/1000)
        # concentration_values.append([V_r, concentration[-1], NernstConcentration])
        phiD_Function = lambda x: spec.barycentricChebInterpolate(D @ potential, x, 0, 1, extrapolation=1)
        phiFunction = lambda x: spec.barycentricChebInterpolate(potential, x, 0, 1, extrapolation=1)
        concentration_Function = lambda x: spec.barycentricChebInterpolate(concentration, x, 0, 1, extrapolation=1)
        j = np.sum(concentration_Function(nodes)*phiD_Function(nodes)*w) + concentration[-1] - concentration[0]
        # anotherJ = (concentration[-1] * np.exp(phi1) - concentration[0] * np.exp(phi0)) / integr.reg_32(
        #                  lambda x: np.exp(z * phiFunction(x)), a=0, b=1, n=2000);
        # print(j - anotherJ)
        # print("j is ", j)
        concentration_values.append([V_r, concentration[-1], -j, NernstConcentration])
        # print("concentraction values are: ", concentration[0], concentration[-1])
        # print(potential[0], potential[-1])
        # x_values = x
        # plt.rcParams['font.family'] = 'Times New Roman'
        # # Plot for potential
        # # Plot for potential
        # plt.figure(figsize=(10, 6))
        # plt.plot(x_values, potential, label='Potential', color='blue')
        # plt.title('Potential Distribution', fontsize=20)  # Increase title font size
        # plt.xlabel('Position', fontsize=16)  # Increase x-axis label font size
        # plt.ylabel('Potential', fontsize=16)  # Increase y-axis label font size
        # plt.grid(True)
        # plt.legend(fontsize=14)  # Increase legend font size
        # plt.tight_layout()
        # plt.show()
        #
        # # Plot for concentration
        # plt.figure(figsize=(10, 6))
        # plt.plot(x_values, concentration, label='Concentration', color='red')
        # plt.title('Concentration Distribution', fontsize=20)  # Increase title font size
        # plt.xlabel('Position', fontsize=16)  # Increase x-axis label font size
        # plt.ylabel('Concentration', fontsize=16)  # Increase y-axis label font size
        # plt.grid(True)
        # plt.legend(fontsize=14)  # Increase legend font size
        # plt.tight_layout()
        # plt.show()
    concentration_values = np.array(concentration_values)
    # print(concentration_values[:, 1])
    # print(concentration_values.shape)
    plt.plot(concentration_values[:, 0], concentration_values[:, 1], color='red', label='Concentration')
    plt.plot(concentration_values[:, 0], concentration_values[:, 2], color='blue',  label='Flux')
    plt.plot(concentration_values[:, 0], concentration_values[:, 3], color='orange',  label='Nernst concentration')
plt.legend(fontsize=14)
plt.grid(True)
plt.show()