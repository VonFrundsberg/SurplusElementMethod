import numpy as np
import scipy.sparse as sparse
import scipy.linalg as sp_linalg
from scipy.interpolate import *
import mathematics.spectral as spectral
import matplotlib.pyplot as plt
import time as time

N = 700; a=-10; b=10
nodes = spectral.chebNodes(n=N, a=a, b=b)
D = spectral.ChebDiffMatrix(n=N, a=a, b=b)
D = np.dot(D, D)
n = 30
V = lambda x: (n)*(n+1)*(2*np.exp(x)/(np.exp(2*x) + 1))**2
# V = lambda x: -x*x
ones = np.sign(nodes)
# ones = np.ones(N)
# ones[:int(N/2)] = -1
M = np.diag(ones)
# print(M)
sol = sp_linalg.eigvals(a=(-D - np.diag(V(nodes)))[1:N, 1:N], b=M[1:N, 1:N])
# print(sol)
plt.scatter(np.real(sol), np.imag(sol))
plt.show()
