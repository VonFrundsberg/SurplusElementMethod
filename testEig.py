import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sp_linalg
from scipy.interpolate import *
from FiniteElementMethod.element.elementOperations import *
from FiniteElementMethod.element.mainElement import *
import FiniteElementMethod.main as fem
import matplotlib.pyplot as plt
import time as time


def schroedinger(N, infN, a, nn):
    gen_obj = fem.FEM()
    msh = mesh(1)
    f = func()
    msh.gen_mesh(np.array([[0.0 - a, a + 0.0]]), n=[nn], p=[N])

    # msh.extendBox(1, 0, [infN])
    msh.file_write('1.txt', '2.txt')
    msh.file_read('1.txt', '2.txt')

    gen_obj.initMesh(msh)

    Ca, Cs = msh.CaCs()
    sigma=msh.sigma1d(Cs)
    # print(Cs)
    F = lambda x: x*x*0 + 1
    n = 30
    # V = lambda x: -(n)*(n+1)*(2*np.exp(x)/(np.exp(2*x) + 1))**2
    V = lambda x: -n*(n+1)/np.cosh(x)**2
    # V = lambda x: -x*2
    # V = lambda x: 0*x - np.pi**2/4
    # V = lambda x: 0*x - 10
    # V = lambda x: 0*x - np.pi**2 + np.sign(x)*1
    # V = lambda x: 0*x - 9*np.pi**2/16
    # V =
    I = lambda u, v, K: f.integr(K=K, elemF=lambda x: f.inner(f.grad(u)(x), f.grad(v)(x)), F=lambda x: F(x)) + \
                        f.integr(K=K, elemF=lambda x: f.inner([u(x)], [v(x)]), F=lambda x: V(x))

    bI1 = lambda u, v, K1, K2: -0.5 * f.integr(K=msh.intersection(K1, K2),
                                                       elemF=lambda x: f.inner(f.orth(f.grad(u)(x), K1, K2), v(x)),
                                                       F=lambda x: F(x)) + \
                                       -0.5 * f.integr(K=msh.intersection(K1, K2),
                                                       elemF=lambda x: f.inner(u(x), f.orth(f.grad(v)(x), K1, K2)),
                                                       F=lambda x: F(x))+ \
                                              f.integr(K=msh.intersection(K1, K2),
                                                       elemF=lambda x: f.inner([u(x)], [v(x)]),
                                                       F=lambda x: F(x)*sigma(x))

    bI2 = lambda u, v, K1, K2: 0.5 * f.integr(K=msh.intersection(K1, K2),
                                                      elemF=lambda x: f.inner(f.orth(f.grad(u)(x), K1, K2), v(x)),
                                                      F=lambda x: F(x)) + \
                                       -0.5 * f.integr(K=msh.intersection(K1, K2),
                                                       elemF=lambda x: f.inner(u(x), f.orth(f.grad(v)(x), K1, K2)),
                                                       F=lambda x: F(x))+ \
                                            -f.integr(K=msh.intersection(K1, K2),
                                                       elemF=lambda x: f.inner([u(x)], [v(x)]),
                                                       F=lambda x: F(x)*sigma(x))


    M = lambda u, v, K: f.integr(K=K, elemF=lambda x: f.inner([u(x)], [v(x)]), F = lambda x: np.sign(x))
    # M = lambda u, v, K: f.integr(K=K, elemF=lambda x: f.inner([u(x)], [v(x)]), F = lambda x: F(x)))

    gen_obj.bilinearForm([I, bI1, bI2])
    gen_obj.rhs(M)
    bc = []
    # bc.append([0, np.inf, 0])
    bc.append([0, -a, 0])
    bc.append([0, a, 0])
    gen_obj.initBC(bc)
    gen_obj.initElems(infMap="linear")
    gen_obj.calcMatrixElemsEig()

    # for i in range(0, 100):
    val, vec = gen_obj.solveEig(k=0, sigma=0 - 1j*1.5)
    # print(val)
    # plt.plot(vec)
    # plt.show()
    gen_obj.fsol = vec
    gen_obj.solToFunc1d()
    G = lambda x: f.calc_func1d(gen_obj.fsol, gen_obj.fLims, x)
    xx = np.linspace(0, a*2, 20000)
    print(gen_obj.eig + 0.5)
    # plt.plot(xx, np.real(G(xx)), color='red')
    # plt.plot(xx, np.imag(G(xx)), color='blue')
    # plt.show()

        # print('hey')
    # time.sleep(500)

for i in range(50, 511):
    # fun(i, i, 10, 2)
    # time.sleep(500)
    # schroedinger(i, i, 5, 4)
    schroedinger(i, i, 10, 11)
# schroedinger(150, 15, 3)
#