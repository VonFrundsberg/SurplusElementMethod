import numpy as np
from mathematics import spectral as spec
from mathematics import integrate as integr
import scipy.linalg as sp
import matplotlib.pyplot as plt
import time as time

n = 80
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
e_r = 20
alpha = f * f * l * l / (R * T * e * e_r) * 100
potential_values = spec.chebNodes(50, 1, 10)
concentration_values = []
for V_r in potential_values:
    # V_r = 90.0
    C0 = 150.0;
    # C1 = 99.0;
    phi0 = 0.0;
    phi1 = F * V_r / 1000 / R / T


    # print(phi1, alpha)
    potential = x.copy()
    concentration = x.copy() + C0
    for i in range(50):
        vectorPhi = potential
        phiFunction = lambda x: spec.barycentricChebInterpolate(vectorPhi, x, 0, 1, extrapolation=1)
        phiD = D @ vectorPhi
        concentrationOperator = -D - z * I * phiD
        concentrationOperator[0, :] = 0
        # concentrationOperator[-1, :] = 0
        concentrationOperator[0, 0] = 1.0
        # concentrationOperator[-1, -1] = 1.0
        j = -(concentration[-1] * np.exp(phi1) - C0 * np.exp(phi0)) / integr.reg_32(
            lambda x: np.exp(z * phiFunction(x)), a=0, b=1, n=1000);
        concentrationRHS = j * np.ones(n)
        concentrationRHS[0] = C0
        # concentrationRHS[-1] = C1
        concentration = sp.solve(concentrationOperator, concentrationRHS)

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
        error = np.max(np.abs(prevPotential - potential))

        # plt.plot(x, concentration)
        # plt.plot(x, potential)
        # plt.show()
    concentration_values.append([error, j, concentration[-1]])
    # print("calculated experimental Nernst Vm: ", R * T / F / z * np.log(concentration[0]/99.0)*1000)
    # print("alpha")
    # print("concentraction values are: ", concentration[0], concentration[-1])
    # # print(potential[0], potential[-1])
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
# plt.plot(potential_values, concentration_values)
#
# plt.show()
errors, js, concentrations = zip(*concentration_values)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(potential_values, errors, marker='o', linestyle='-', color='b', label='Errors')
plt.plot(potential_values, js, marker='s', linestyle='--', color='r', label='js')
plt.plot(potential_values, concentrations, marker='^', linestyle='-.', color='g', label='Concentrations')

# Add labels and title
plt.xlabel('Potential Values')
plt.ylabel('Values')
plt.title('Errors, js, and Concentrations vs. Potential with C0 = {}'.format(C0))
plt.grid(True)
plt.legend()

# Show plot
plt.tight_layout()
plt.show()