import scipy.linalg as sp_lin
import numpy as np

def newtonIterationHybrid(G, L, F, tol, u0):
    uPrev = 0
    uNext = u0
    while sp_lin.norm(uPrev - uNext) > tol:
        D = G * uPrev + L
        invD = sp_lin.inv(D)
        uNext = uPrev - invD * F(uPrev)