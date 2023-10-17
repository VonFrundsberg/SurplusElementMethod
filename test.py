import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sp_linalg
import FiniteElementMethod.mesh.mesh as MeshClass
import FiniteElementMethod.main as fem
import matplotlib.pyplot as plt
import time as time

def fun(N, infN, a, nn):
    finiteElementObject = fem.FEM()

    gradForm = "integral w(x) grad(u).grad(v)"
    boundaryForm1 = "boundaryIntegral w(x) [u] <grad(v).n>"
    boundaryForm2 = "boundaryIntegral w(x) [v] <grad(u).n>"
    functional = "integral w(x) u f"

    finiteElementObject.setBilinearForm([gradForm, boundaryForm1, boundaryForm2])
    finiteElementObject.setRHSFunctional(functional)

    boundaryConditions = "axis: 1, boundary: right, value: 0.0"
    finiteElementObject.setDirichletBoundaryConditions(boundaryConditions)

    mesh = MeshClass.mesh(1)
    mesh.generateUniformMeshOnRectange([0, a], nn, N)
    mesh.establishNeighbours()
    mesh.fileWrite("elementsData.txt", "neighboursData.txt")
    mesh.fileRead("elementsData.txt", "neighboursData.txt")
    finiteElementObject.initializeMesh(mesh)
    finiteElementObject.initializeElements()
    print('done')
    time.sleep(500)
    # msh.extendBox(1, 0, [infN])
    # msh.file_write('1.txt', '2.txt')
    # msh.file_read('1.txt', '2.txt')
    # gen_obj.initMesh(msh)
    # # msh.CtCg()
    # Ca, Cs = msh.CaCs()
    # print(Ca, Cs)
    # s_list = np.array(msh.sigma_list(Cs))*(10**2)
    # print("s list ", s_list)
    # bI1 = lambda u, v, K1, K2: -0.5 * f.integr(K=msh.intersection(K1, K2),
    #                                           f.integr(K=msh.intersection(K1, K2),
    #                                                    elemF=lambda x: f.inner([u(x)], [v(x)]),
    #                                                    F=lambda x: F(x)*sigma(x)))
    # print(Ca, Cs)
    I = gen_obj.laplaceOperator(dim=1, F=lambda x: x*x, sigma=msh.sigma1d(Cs))
    # F = lambda x: x*x*np.exp(-(x - 10))*np.heaviside(x - 10, 1/2)*(x - 10)**2
    # F = lambda x: -x*x*np.exp(-x)
    F = lambda x: -x*(np.exp(-x))*(2 + (-2 + x)*x)
    fI = lambda v, K: f.integr(K=K, elemF=lambda x: [v(x)], F=F, n=1000)

    gen_obj.bilinearForm(I)
    gen_obj.rhs(fI)

    bc = []
    bc.append([0, np.inf, 0])
    gen_obj.initBC(bc)


    gen_obj.initElems(infMap="linear")
    gen_obj.calcMatrixElems()
    gen_obj.solve()

    # asol = lambda x: np.exp(-x) + 2*np.exp(-x)/x - 2/x
    asol = lambda x: np.exp(-x)*(x)
    # asol = lambda x: np.abs(26 + np.heaviside(x - 10, 1/2)*(-26 - 38*np.exp(10-x) + 344/x -\
    #                                                     104*np.exp(10-x)/x + 14* np.exp(10-x)*x - np.exp(10 - x)*x*x))
    # a = np.loadtxt("example60.txt")
    # gen_obj.sol[0][:N] = -a[:N]
    # gen_obj.sol[1][:-1] = -a[N:]
    # print(a.shape)
    # print(gen_obj.sol[0].shape)
    # gen_obj.sol =
    gen_obj.solToFunc1d()
    G = lambda x: f.calc_func1d(gen_obj.fsol, gen_obj.fLims, x)

    error = 0
    for it in msh:
        tmp = b_elem(it[0][0], 2)
        w, x = intg.reg_32_wn(-1, 1, 1000)
        w /= tmp.dmap(x)
        x = tmp.map(x)
        # print(np.max(np.abs(asol(x) - G(x))))
        plt.loglog(x, np.abs(G(x) + asol(x)))
        gx = G(x)
        # plt.loglog(x, w*x**2/(x + 1)**4*(np.nan_to_num(gx, 0) + asol(x))**2)
        # print(G(x))

        # error += np.sum(w*x**2/(x + 1)**4*(np.nan_to_num(gx, 0) + asol(x))**2)
        error += np.sum(w*x*x*(np.nan_to_num(gx, 0) + asol(x))**2)
        # error += np.sum(w*x**2/(x + 1)**4*(asol(x))**2)
    print(N, error)
    # time.sleep(1)
    plt.show()
    # j = 0
    # vec = []
    # print(N, np.abs(gen_obj.sol[0][0] + 1))

    # for it in msh:
    #     tmp = b_elem(it[0][0], 2)
    #     x = sp.chebNodes(N)
    #     # M = gen_obj.getA()
    #
    #     xs = tmp.map(x)
    #     # print(np.max(np.abs(asol(x) - G(x))))
    #     # print([gen_obj.sol[j][0]])
    #     # print([gen_obj.sol[j][-1]])
    #     # plt.plot(x, asol(xs) - gen_obj.sol[j])
    #     j += 1
    #     # error += np.sum(w*x**2/(x + 1)**4*(asol(x) - G(x))**2)
    #     # error += np.sum(w*x**2/(x + 1)**4*(asol(x))**2)
    #     # plt.show()


    return np.sqrt(error)/0.335295


def fun2d(N, infN, a, nn):
    gen_obj = fem.FEM()
    msh = mesh(2)
    f = func()
    msh.gen_mesh(np.array([[0, a], [-a, a]], dtype=np.float), n=[2, 2], p=[N, N])
    msh.customExtendBox([0, 1], [infN, infN])
    msh.file_write('1.txt', '2.txt')
    msh.file_read('1.txt', '2.txt')
    gen_obj.initMesh(msh)
    z_0 = 0
    I = gen_obj.laplaceOperator(dim=2, F=lambda x, y: x, sigma=lambda x, y: x*0 + 1)
    F = lambda x, y: -x*np.exp(-np.sqrt(x*x + (y - z_0)**2))
    # F = lambda x, y: np.cos(y)*(np.cos(x) - 2*x*np.sin(x))
    fI = lambda v, K: f.integr(K=K, elemF=lambda x: [v(x)], F=F)
    # print("hey")
    gen_obj.bilinearForm(I)
    gen_obj.rhs(fI)

    bc = []
    bc.append([0, np.inf, 0])
    bc.append([1, -np.inf, 0])
    bc.append([1, np.inf, 0])

    # bc.append([0, 15, 0])
    # bc.append([1, 0, 0])
    # bc.append([1, 30, 0])

    gen_obj.initBC(bc)
    gen_obj.initElems(infMap='linear')

    gen_obj.calcMatrixElems()
    # print('hey')
    gen_obj.solve()
    gen_obj.solToFunc2d()
    gen_obj.plot2d()
    # G = lambda x: f.calc_func2d(gen_obj.fsol, gen_obj.fLims, x)
    # x = np.linspace(0, a, 100)
    # y = np.linspace(-a, a, 200)
    # plt.imshow(G(approx.meshgrid(x, y)))
    # nsol = G(approx.meshgrid(x, y))
    # plt.show()
    # xx, yy = approx.meshgrid(x, y)
    # asol = lambda x, y: np.exp(-np.sqrt(x**2 + (y - z_0)**2)) +\
    #                     2*np.exp(-np.sqrt(x**2 + (y - z_0)**2))/np.sqrt(x**2 + (y - z_0)**2) - 2/np.sqrt(x**2 + (y - z_0)**2)

    # asol = lambda x, y: np.sin(x)*np.cos(y)
    # for i in range():
    #     print(it)
    #     plt.imshow(it)

    plt.show()
    # print(np.max(np.abs(asol(xx, yy) + nsol)))
    # diff = asol(xx, yy) + nsol
    # print(np.max(np.abs(diff)))
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    from mpl_toolkits.mplot3d import Axes3D
    # ax = fig.gca(projection='3d')
    # ax.plot_surface(xx, yy, diff)
    # ax.plot_surface(xx, yy, asol(xx, yy))
    # plt.show()
    # plt.pcolor(xx, yy, asol(xx, yy) + nsol)
    # plt.show()
    # print('hello')
    # time.sleep(500)
    # asol = lambda x: np.exp(-x) + 2*np.exp(-x)/x - 2/x
    # asol = lambda x: np.abs(26 + np.heaviside(x - 10, 1/2)*(-26 - 38*np.exp(10-x) + 344/x -\
    #                                                     104*np.exp(10-x)/x + 14* np.exp(10-x)*x - np.exp(10 - x)*x*x))
    # a = np.loadtxt("example60.txt")
    # gen_obj.sol[0][:N] = -a[:N]
    # gen_obj.sol[1][:-1] = -a[N:]
    # print(a.shape)
    # print(gen_obj.sol[0].shape)
    # gen_obj.sol =
    # gen_obj.solToFunc1d()
    # G = lambda x: f.calc_func1d(gen_obj.fsol, gen_obj.fLims, x)
    #
    # error = 0
    # for it in msh:
    #     tmp = b_elem(it[0][0], 2)
    #     w, x = intg.reg_32_wn(-1, 1, 5000)
    #     w /= tmp.dmap(x)
    #     x = tmp.map(x)
    #     # print(np.max(np.abs(asol(x) - G(x))))
    #     # plt.loglog(x, G(x) + asol(x))
    #     error += np.sum(w*x**2/(x + 1)**4*(G(x) + asol(x))**2)
    #     # error += np.sum(w*x**2/(x + 1)**4*(asol(x))**2)
    # print('done')
    return 0
    # return np.sqrt(error)/13.9332629589721

def schroedinger(N, a, nn):
    gen_obj = fem.FEM()
    msh = mesh(1)
    f = func()
    msh.gen_mesh(np.array([[-a + 0.0, a + 0.0]]), n=[nn], p=[N])

    # msh.extendBox(1, 0, [infN])
    msh.file_write('1.txt', '2.txt')
    msh.file_read('1.txt', '2.txt')

    gen_obj.initMesh(msh)

    Ca, Cs = msh.CaCs()
    sigma=msh.sigma1d(0)
    F = lambda x: x*0 + 1
    n = 5
    # V = lambda x: -(n)*(n+1)*(2*np.exp(x)/(np.exp(2*x) + 1))**2
    V = lambda x: x**2
    # V = lambda x: 0*x - np.pi**2/4
    # V = lambda x: 0*x - 10
    # V = lambda x: 0*x - np.pi**2 + np.sign(x)*1
    # V = lambda x: 0*x - 9*np.pi**2/16
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


    # M = lambda u, v, K: f.integr(K=K, elemF=lambda x: f.inner([u(x)], [v(x)]), F = lambda x: np.sign(x))
    M = lambda u, v, K: f.integr(K=K, elemF=lambda x: f.inner([u(x)], [v(x)]), F = lambda x: x*0 + 1)

    gen_obj.bilinearForm([I, bI1, bI2])
    gen_obj.rhs(M)
    bc = []
    bc.append([0, a, 0])
    bc.append([0, -a, 0])
    # bc.append([0, a, 0])
    gen_obj.initBC(bc)
    gen_obj.initElems()
    gen_obj.calcMatrixElemsEig()

    for i in range(0, 100):
        gen_obj.solveEig(k=1, sigma=0 - 1j*1.5)
        gen_obj.solToFunc1d()
        G = lambda x: f.calc_func1d(gen_obj.fsol, gen_obj.fLims, x)
        xx = np.linspace(-a, a, 20000)
        # print(gen_obj.eig)
        plt.plot(xx, np.real(G(xx)), color='red')
        plt.plot(xx, np.imag(G(xx)), color='blue')
        plt.show()

        # print('hey')
    time.sleep(500)

for i in range(2, 200):
    fun(i, i, 5, 2)
    # fun2d(i, i, 5, 2)
    # time.sleep(500)
# schroedinger(150, 15, 3)
