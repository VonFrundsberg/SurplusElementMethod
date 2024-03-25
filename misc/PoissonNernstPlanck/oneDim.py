import numpy as np
from mathematics import spectral as spec
from mathematics import integrate as integr
import scipy.linalg as sp
import matplotlib.pyplot as plt

n = 20
#WORKS ONLY FOR Z = 1
z = 1
D = spec.chebDiffMatrix(n, 0, 1)
I = np.eye(n)
x = spec.chebNodes(n, 0, 1)

C0 = 1.0; C1 = 1.5; phi0 = 0.0; phi1 = 1.0
potential = x

for i in range(50):
    vectorPhi = potential
    phiFunction = lambda x: spec.barycentricChebInterpolate(vectorPhi, x, 0, 1, extrapolation=1)
    phiD = D @ vectorPhi
    concentrationOperator = -D - z * I * phiD
    concentrationOperator[0, :] = 0
    # concentrationOperator[-1, :] = 0
    concentrationOperator[0, 0] = 1.0
    # concentrationOperator[-1, -1] = 1.0

    concentrationRHS = -(C1*np.exp(phi1) - C0 * np.exp(phi0))/integr.reg_32(lambda x: np.exp(z * phiFunction(x)), a=0, b=1, n=1000) * np.ones(n)
    concentrationRHS[0] = C0
    # concentrationRHS[-1] = C1
    concentration = sp.solve(concentrationOperator, concentrationRHS)

    potentialOperator = D @ D
    potentialOperator[0, :] = 0
    potentialOperator[-1, :] = 0
    potentialOperator[0, 0] = 1.0
    potentialOperator[-1, -1] = 1.0
    potentialRHS = -z * concentration
    potentialRHS[0] = phi0
    potentialRHS[-1] = phi1
    potential = sp.solve(potentialOperator, potentialRHS)

    # plt.plot(x, concentration)
    # plt.plot(x, potential)
    # plt.show()
x_values = x
plt.rcParams['font.family'] = 'Times New Roman'
# Plot for potential
# Plot for potential
plt.figure(figsize=(10, 6))
plt.plot(x_values, potential, label='Potential', color='blue')
plt.title('Potential Distribution', fontsize=20)  # Increase title font size
plt.xlabel('Position', fontsize=16)  # Increase x-axis label font size
plt.ylabel('Potential', fontsize=16)  # Increase y-axis label font size
plt.grid(True)
plt.legend(fontsize=14)  # Increase legend font size
plt.tight_layout()
plt.show()

# Plot for concentration
plt.figure(figsize=(10, 6))
plt.plot(x_values, concentration, label='Concentration', color='red')
plt.title('Concentration Distribution', fontsize=20)  # Increase title font size
plt.xlabel('Position', fontsize=16)  # Increase x-axis label font size
plt.ylabel('Concentration', fontsize=16)  # Increase y-axis label font size
plt.grid(True)
plt.legend(fontsize=14)  # Increase legend font size
plt.tight_layout()
plt.show()