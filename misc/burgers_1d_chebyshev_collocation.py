import numpy as np
import mathematics.spectral as spec
import scipy.linalg as sp_lin
import matplotlib.pyplot as plt
import mathematics.nonlinear as nonlin


n = 200
chebPoints = spec.chebNodes(n, -1, 1)
D = spec.chebDiffMatrix(n, -1, 1)
D2 = D @ D
dt = 1e-3
uPrev = np.ones(n)
v = 0.01
t0 = 1.0
a = 16
phi = lambda x, t: x/t * (np.sqrt(a/t)*np.exp(-x**2/(4*v*t)))/(1 + np.sqrt(a/t)*np.exp(-x**2/(4*v*t)))
asol = lambda x, t, c: c + phi((x - c*t), t + t0)
t = np.linspace(0, 1, 100)

plt.plot(chebPoints, asol(chebPoints, 0, 1))
plt.plot(chebPoints, asol(chebPoints, 1, 1))
plt.show()

u0_func = lambda x: 
uNext = nonlin.newtonIterationHybrid(1/2*D, v*D2, 1/2*D + v*D2)


