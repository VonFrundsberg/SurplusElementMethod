import numpy as np
import mathematics.spectral as spec
import scipy.linalg as sp_lin
import matplotlib.pyplot as plt
import mathematics.nonlinear as nonlin


n = 200
chebPoints = spec.chebNodes(n, -1, 1)
D = spec.chebDiffMatrix(n, -1, 1)
D2 = D @ D
dt = 1e-2
uPrev = np.ones(n)
v = 0.01
t0 = 1.0
a = 16
c = 1
phi = lambda x, t: x/t * (np.sqrt(a/t)*np.exp(-x**2/(4*v*t)))/(1 + np.sqrt(a/t)*np.exp(-x**2/(4*v*t)))
asol = lambda x, t: c + phi((x - c*t), t + t0)

plt.plot(chebPoints, asol(chebPoints, 0))
plt.plot(chebPoints, asol(chebPoints, 1))
plt.show()

u0_func = lambda x: asol(x, 0)
uLeft = lambda t: asol(-1, t)
uRight = lambda t: asol(1, t)
uPrev = u0_func(chebPoints)
# plt.plot(chebPoints, uPrev)
# plt.show()
curT = 0
for i in range(100):
    uNext = uPrev + (-(uPrev * (D @ uPrev)) + v * (D2 @ uPrev)) * dt
    curT += dt
    uNext[0] = uLeft(curT)
    uNext[-1] = uRight(curT)
    plt.plot(chebPoints, uNext - asol(chebPoints, curT))
    plt.show()
    uPrev = uNext

plt.plot(chebPoints, uNext)
plt.show()


