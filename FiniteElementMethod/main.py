import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sp_linalg
import scipy.linalg as sp_lin
from FiniteElementMethod.element import mainElementClass as element
import time as time
class FEM:

    def setBilinearForm(self, bilinearFormsList):
        self.innerForms = [bilinearFormsList[0]]
        self.boundaryForms = [bilinearFormsList[1], bilinearFormsList[2]]

        return None

    def setRHSFunctional(self, functional):
        self.functional = functional
        return None
    def setDirichletBoundaryConditions(self, boundaryConditions):
        return None
    def initializeMesh(self, mesh):
        """Set up already made rectangular mesh, which is an object of SurplusElementMethod/FiniteElementMethod/mesh class

        Arguments:
        mesh: list of 2 objects [mesh.elements, mesh.neighbours]. Elements contain info about domain decomposition and
        order of polynomial approximation, neighbours contain info about neighbouring elements. More information in the
        corresponding class

        Returns:
        Nothing, creates self.mesh field in FEM class
        """
        self.mesh = mesh

    def initializeElements(self):
        """
        """
        elementsAmount = self.mesh.getElementsAmount()
        self.elements = [None] * elementsAmount
        for i in range(elementsAmount):
            tmpElement = self.mesh.elements[i]
            self.elements[i] = element.element(tmpElement[:, :2], polynomialOrder=tmpElement[:, -2],
                                               mappingType=tmpElement[-1])

    def calculateElements(self):
        """
        For each element in self.mesh, calculates its discretized version,
         using previously initialized bilinearForms, and RHS functional
                """
        elementsAmount = self.mesh.getElementsAmount()

        self.matrixElements = [None] * elementsAmount
        for i in range(elementsAmount):
            self.matrixElements[i] = [None] * elementsAmount
        self.functionalElements = [None] * elementsAmount

        innerFormsAmount = len(self.innerForms)
        boundaryFormsAmount = len(self.boundaryForms)

        for i in range(elementsAmount):
            innerMatrix = self.innerForms[0](self.elements[i], self.elements[i])

            for j in range(1, innerFormsAmount):
                innerMatrix += self.innerForms[0](self.elements[i], self.elements[i])

            # innerMatrix = self.mreshape(self.mesh[i][1], self.mesh[i][1], innerMatrix)
            self.matrixElements[i][i] = sparse.csr_matrix(innerMatrix)

            # self.functionalElements[i] = (self.functional(self.elements[i])).flatten()

            print(str(i) + ' \'s element calculated')

            for neighborNumber in self.mesh.neighbours[i]:
                # K1 = self.mesh[i][0]
                # K2 = self.mesh[it[1]][0]

                # self.matrixElems[i][i] += self.mreshape(self.mesh[i][1], self.mesh[i][1],
                #                                         self.bform1(self.elems[i], self.elems[i], K1=K1, K2=K2))
                for boundaryForm in self.boundaryForms:
                    self.matrixElements[i][i] += boundaryForm(self.elements[i], self.elements[i])

                if i < it[1]:
                    self.matrixElems[i][it[1]] = sparse.csr_matrix(self.mreshape(self.mesh[i][1], self.mesh[it[1]][1],
                                                            self.bform2(self.elems[i], self.elems[it[1]], K1=K1, K2=K2)))
                else:
                    self.matrixElems[i][it[1]] = self.matrixElems[it[1]][i].T

    def solveSLAE(self):
        return None

    def solutionToFunctionFormat(self):
        return None


    def initElems(self, infMap):
        N = len(self.mesh.l)
        self.elems = [None] * N
        for i in range(N):
            x = self.mesh[i][0]
            args = [None] * self.mesh.n

            for j in range(len(self.bc)):

                arg = np.argwhere(self.mesh[i][0][self.bc[j][0], :] == self.bc[j][1])


                if len(arg) > 0:
                    if args[self.bc[j][0]] is not None:
                        args[self.bc[j][0]].append(np.hstack([-np.squeeze(arg), self.bc[j][2]]))
                    else:
                        args[self.bc[j][0]] = [np.hstack([-np.squeeze(arg), self.bc[j][2]])]

            args = np.array(args)
            self.elems[i] = FEM.elem(*self.mesh[i], bc=args, infMap=infMap)

    def bilinearForm(self, I):
        self.iform = I[0]
        self.bform1 = I[1]
        self.bform2 = I[2]

    def rhs(self, f):
        self.fform = f

    def initBC(self, bc):
        self.bc = bc
    def mreshape(self, s1, s2, tensor):
        shape = np.hstack([s1, s2])
        sort = 0
        if shape.size == 2:
            shape = shape[[0, 1]]
            sort = np.argsort([0, 1])
        if shape.size == 4:
            shape = shape[[0, 2, 1, 3]]
            sort = np.argsort([0, 2, 1, 3])
        if shape.size == 6:
            shape = shape[[0, 3, 1, 4, 2, 5]]
            sort = np.argsort([0, 3, 1, 4, 2, 5])

        tensor = np.reshape(tensor, shape)
        tensor = np.transpose(tensor, sort)
        tensor = np.reshape(tensor, [np.prod(s1), np.prod(s2)])
        return tensor

    def laplaceOperator(self, dim, F, sigma = lambda x: x*0):
        if dim == 1:
            f = func()
            msh = mesh(dim)
            I = lambda u, v, K: f.integr(K=K, elemF=lambda x: f.inner(f.grad(u)(x), f.grad(v)(x)), F=lambda x: F(x)) -\
                                f.integr(K=K, elemF=lambda x: f.inner([u(x)], [v(x)]), F=lambda x: 2*x)
            # I = lambda u, v, K: f.integr2(K=K, elemF=lambda x: f.inner(f.grad(u)(x), f.grad(v)(x)), F=lambda x: F(x))

            bI1 = lambda u, v, K1, K2: -0.5 * f.integr(K=msh.intersection(K1, K2),
                                                       elemF=lambda x: f.inner(f.orth(f.grad(u)(x), K1, K2), v(x)),
                                                       F=lambda x: F(x)) + \
                                       -0.5 * f.integr(K=msh.intersection(K1, K2),
                                                       elemF=lambda x: f.inner(u(x), f.orth(f.grad(v)(x), K1, K2)),
                                                       F=lambda x: F(x)) + \
                                              f.integr(K=msh.intersection(K1, K2),
                                                       elemF=lambda x: f.inner([u(x)], [v(x)]),
                                                       F=lambda x: F(x)*sigma(x))
            self.bform3 = lambda u, v, K1, K2: f.integr(K=msh.intersection(K1, K2),
                                                       elemF=lambda x: f.inner([u(x)], [v(x)]),
                                                       F=lambda x: F(x)*sigma(x))

            bI2 = lambda u, v, K1, K2: 0.5 * f.integr(K=msh.intersection(K1, K2),
                                                      elemF=lambda x: f.inner(f.orth(f.grad(u)(x), K1, K2), v(x)),
                                                      F=lambda x: F(x)) + \
                                       -0.5 * f.integr(K=msh.intersection(K1, K2),
                                                       elemF=lambda x: f.inner(u(x), f.orth(f.grad(v)(x), K1, K2)),
                                                       F=lambda x: F(x)) + \
                                            -f.integr(K=msh.intersection(K1, K2),
                                                       elemF=lambda x: f.inner([u(x)], [v(x)]),
                                                       F=lambda x: F(x)*sigma(x))
            return I, bI1, bI2
        if dim == 2:
            f = func()
            msh = mesh(dim)
            I = lambda u, v, K: f.integr(K=K, elemF=lambda x: f.inner(f.grad(u)(x), f.grad(v)(x)), F=lambda x, y: F(x, y))

            bI1 = lambda u, v, K1, K2: -0.5 * f.integr(K=msh.intersection(K1, K2),
                                                       elemF=lambda x: f.inner(f.orth(f.grad(u)(x), K1, K2), v(x)),
                                                       F=lambda x, y: F(x, y)) + \
                                       -0.5 * f.integr(K=msh.intersection(K1, K2),
                                                       elemF=lambda x: f.inner(u(x), f.orth(f.grad(v)(x), K1, K2)),
                                                       F=lambda x, y: F(x, y)) #+ \
                                               # f.integr(K=msh.intersection(K1, K2),
                                               #          elemF=lambda x: f.inner([u(x)], [v(x)]),
                                               #          F=lambda x, y: F(x, y)*sigma(x))

            bI2 = lambda u, v, K1, K2: 0.5 * f.integr(K=msh.intersection(K1, K2),
                                                      elemF=lambda x: f.inner(f.orth(f.grad(u)(x), K1, K2), v(x)),
                                                      F=lambda x, y: F(x, y)) + \
                                       -0.5 * f.integr(K=msh.intersection(K1, K2),
                                                       elemF=lambda x: f.inner(u(x), f.orth(f.grad(v)(x), K1, K2)),
                                                       F=lambda x, y: F(x, y)) #+ \
                                             # -f.integr(K=msh.intersection(K1, K2),
                                             #            elemF=lambda x: f.inner([u(x)], [v(x)]),
                                             #            F=lambda x, y: F(x, y)*sigma(x))
            return I, bI1, bI2



    def nonZero(self):
        A = sparse.bmat(self.matrixElems)
        A = sparse.csr_matrix(A)
        return A.count_nonzero()

    def calcMatrixElems(self):
        N = len(self.mesh.l)
        self.matrixElems = [None] * N
        self.rhs = [None] * N
        for i in range(N):
            self.matrixElems[i] = [None] * N
        self.something = []
        for i in range(N):
            tmp1 = self.iform(self.elems[i], self.elems[i], K=self.mesh[i][0])
            tmp1 = self.mreshape(self.mesh[i][1], self.mesh[i][1], tmp1)

            self.matrixElems[i][i] = sparse.csr_matrix(tmp1)
            # print(self.matrixElems[i][i][0, 0])

            self.rhs[i] = (self.fform(self.elems[i], K=self.mesh[i][0])).flatten()
            print(str(i) + ' \'s element calculated')
            for it in self.mesh.neigh[i]:
                K1 = self.mesh[i][0]
                K2 = self.mesh[it[1]][0]
                # self.something.append(self.matrixElems[i][i] + self.mreshape(self.mesh[i][1], self.mesh[i][1],
                #                                         self.bform3(self.elems[i], self.elems[i], K1=K1, K2=K2)))
                self.matrixElems[i][i] += self.mreshape(self.mesh[i][1], self.mesh[i][1],
                                                        self.bform1(self.elems[i], self.elems[i], K1=K1, K2=K2))


                if i < it[1]:
                    self.matrixElems[i][it[1]] = sparse.csr_matrix(self.mreshape(self.mesh[i][1], self.mesh[it[1]][1],
                                                            self.bform2(self.elems[i], self.elems[it[1]], K1=K1, K2=K2)))
                    # np.set_printoptions(precision=3)
                    # print(self.mreshape(self.mesh[i][1], self.mesh[it[1]][1],
                    #                                         self.bform2(self.elems[i], self.elems[it[1]], K1=K1, K2=K2)))
                else:
                    self.matrixElems[i][it[1]] = self.matrixElems[it[1]][i].T
    def getA(self):
        A = sparse.bmat(self.matrixElems)
        A = sparse.csr_matrix(A)
        return A
    def solve(self):

        A = sparse.bmat(self.matrixElems)
        # np.set_printoptions(precision=6, suppress=True)
        # print(self.matrixElems[0][0])
        # print(self.matrixElems[0][1].toarray())
        # print(self.matrixElems[1][0].toarray())
        # print(self.matrixElems[1][1])
        import time as time
        A = sparse.csr_matrix(A)
        # np.set_printoptions(precision=5, suppress=True)
        # aarr = A.toarray()
        # print('hey')
        # import numpy.linalg as np_linalg
        # print("eigs", np_linalg.eigvals(aarr))
        # print("cond", np_linalg.cond(aarr))
        # time.sleep(500)
        # print(aarr)
        ind = (A.getnnz(1) > 0).copy()

        A = A[A.getnnz(1) > 0, :][:, A.getnnz(0) > 0]
        # aarr = A.toarray()
        # print("diag", np.diag(aarr))
        # flag = 0
        # for it in self.something:
        #     print(np.array(1/2*(np.diag(it)) - np.diag(aarr)[flag:3 + flag], dtype=np.float))
        #     flag = 3
        # time.sleep(500)
        self.rhs = np.hstack(self.rhs)
        # print(self.rhs)

        self.rhs = self.rhs[ind]

        sol = sp_linalg.spsolve(A, -self.rhs)
        # print(sol)

        # time.sleep(500)
        # print(sol)
        sols = []
        grids = []
        i1 = 0
        for i in range(len(self.mesh.l)):
            ps = self.mesh[i][1]
            bcs, pad = self.elems[i].bcs()
            ps = np.array(ps - bcs, dtype=np.int64)
            i2 = i1 + np.prod(ps)
            tmp = sol[i1: i2]
            tmp = np.reshape(tmp, ps)
            tmp = np.pad(tmp, pad, 'constant')
            sols.append(tmp)
            grids.append(self.elems[i].grid)
            # print(self.mesh[i])
            i1 = i2
        self.grid = grids
        self.sol = sols

    def calcMatrixElemsEig(self):
        N = len(self.mesh.l)
        self.matrixElems = [None] * N
        self.rhs = [None] * N
        for i in range(N):
            self.matrixElems[i] = [None] * N
            self.rhs[i] = [None] * N

        for i in range(N):
            tmp1 = self.iform(self.elems[i], self.elems[i], K=self.mesh[i][0])
            tmp1 = self.mreshape(self.mesh[i][1], self.mesh[i][1], tmp1)
            self.matrixElems[i][i] = tmp1

            tmp1 = self.fform(self.elems[i], self.elems[i], K=self.mesh[i][0])
            tmp1 = self.mreshape(self.mesh[i][1], self.mesh[i][1], tmp1)
            self.rhs[i][i] = tmp1

            for it in self.mesh.neigh[i]:
                K1 = self.mesh[i][0]
                K2 = self.mesh[it[1]][0]
                self.matrixElems[i][i] += self.mreshape(self.mesh[i][1], self.mesh[i][1],
                                                        self.bform1(self.elems[i], self.elems[i], K1=K1, K2=K2))

                if i < it[1]:
                    self.matrixElems[i][it[1]] = self.mreshape(self.mesh[i][1], self.mesh[it[1]][1],
                                                               self.bform2(self.elems[i], self.elems[it[1]], K1=K1,
                                                                           K2=K2))
                else:
                    self.matrixElems[i][it[1]] = self.matrixElems[it[1]][i].T

    def solveEig(self, sigma=0, k=0):
        # t = time.time()
        # A = sparse.bmat([[sparse.csr_matrix(self.matrixElems[0][0]), self.matrixElems[0][1]],
        #                 [self.matrixElems[1][0], self.matrixElems[1][1]]])
        A = sparse.bmat(self.matrixElems)
        A = sparse.csr_matrix(A)
        A = A[A.getnnz(1) > 0, :][:, A.getnnz(0) > 0]
        import matplotlib.pyplot as plt
        # plt.imshow(np.real())
        # plt.show()
        # print(self.rhs[-1][-1])
        # print(np.max(np.abs(self.rhs[-1][-1])))
        M = sparse.bmat(self.rhs)
        M = sparse.csr_matrix(M)
        M = M[M.getnnz(1) > 0, :][:, M.getnnz(0) > 0]

        # u, s, v = sp_lin.svd(((A.dot(M)).todense()))
        # print(s)
        import time as time
        # time.sleep(500)
        # eig, sol = sp_linalg.eigs(A=A, M=M, k=k, sigma=sigma)
        # print(eig)
        A = A.todense(); M = M.todense()
        # eig, sol =
        # eigs = sp_lin.eigvals(A, M)
        # eigs = np.sort(eigs)
        # print(A.shape)
        # f = lambda x: sp_lin.det(A - x*M)
        # a = -10; b = 10
        # nodes = sp.chebNodes(400, a=a, b=b) + 3.42671030j
        # xs = np.linspace(a, b, 1000) + 3.42671030j
        # fc = np.array(list(map(f, xs)))
        # plt.plot(xs, np.real(fc), 'red')
        # xs = np.linspace(a, b, 1000) + 3.42671030j
        # fc = np.array(list(map(f, xs)))
        # plt.plot(xs, np.imag(fc), 'green')
        # xs = np.linspace(a, b, 1000) + 3j
        # fc = np.array(list(map(f, xs)))
        # plt.plot(xs, np.abs(fc), 'green')
        # xs = np.linspace(a, b, 1000) + 4j
        # fc = np.array(list(map(f, xs)))
        # plt.plot(xs, np.abs(fc), 'black')
        # xs = np.linspace(a, b, 1000)
        # fc = np.array(list(map(f, xs)))
        # plt.plot(xs, fc, 'blue')

        # nodes = sp.chebNodes(400, a=a, b=b) + 3j
        # xs = np.linspace(a, b, 20000) + 3j
        # fc = sp.bary(np.array(list(map(f, nodes))), xs, a=a,b=b, cx=nodes)
        # plt.plot(xs, np.abs(fc), 'green')
        #
        # nodes = sp.chebNodes(400, a=a, b=b)
        # # print(nodes)
        # xs = np.linspace(a, b, 20000)
        # fc = sp.bary(np.array(list(map(f, nodes))), xs, a=a,b=b, cx=nodes)
        # plt.plot(xs, np.abs(fc), 'blue')
        # plt.show()
        # print(A.shape)
        # print(M.shape)
        eigval, eigvec = sp_lin.eig(A, M)
        # print(sp_lin.eigvals(A))
        ga1 = np.argwhere(np.abs(np.imag(eigval)) < 1e+2)
        ga2 = np.argwhere(np.abs(np.real(eigval)) < 2*1e+3)
        # ga3 = np.argwhere(np.abs(np.imag(eigval)) > 1e-2)
        ga = np.intersect1d(ga1, ga2)
        # ga = np.intersect1d(ga, ga3)
        print(ga.size)
        plt.scatter(np.real(eigval[ga]), np.imag(eigval[ga]))
        plt.show()
        # print(*nonreal)
        # k = 52
        # sol = eigvec[:, nonreal][:, k]
        # print(eigval[nonreal][k])
        import matplotlib.pyplot as plt
        import matplotlib.pyplot as plt
        # plt.imshow(np.real(eigvec))
        # plt.show()
        # plt.imshow(np.imag(eigvec))
        # plt.show()
        # plt.show()
        # for it in nonreal:
        #     print(eigval[it])
        #     plt.plot(np.imag(eigvec[:, it]))
        #     plt.plot(np.real(eigvec[:, it]))
        #     plt.show()
        # print(eigval)

        # sortedargs = np.argsort((np.real(eigval)))
        # eigval = eigval[sortedargs]
        # eigvec = eigvec[:, sortedargs]
        # # print(eigval)
        # plt.plot(eigval)
        # plt.show()
        sols = []
        i1 = 0
        sol = eigvec[:, k]
        eig = eigval[k]
        for i in range(len(self.mesh.l)):
            ps = self.mesh[i][1]
            bcs, pad = self.elems[i].bcs()
            ps = np.array(ps - bcs, dtype=np.int)
            i2 = i1 + np.prod(ps)
            tmp = sol[i1: i2]
            tmp = np.reshape(tmp, ps)
            tmp = np.pad(tmp, pad, 'constant')
            sols.append(tmp)
            i1 = i2
        self.sol = sols
        self.eig = eig
        # print(eigvec[:, 0])
        return eigval[0], eigvec[:, 0]

    def solToFunc1d(self):
            self.fsol = []
            self.fLims = np.zeros([2, len(self.elems)])
            for i in range(len(self.elems)):
                xs = np.squeeze(np.array(self.elems[i].grid()))
                self.fLims[:, i] = np.array([xs[0], xs[-1]])
            f = lambda i, x: barycentric_interpolate(self.elems[i].new_grid()[0][0],
                                                     self.sol[i].copy(), self.elems[i].new_grid()[0][1](x))
            self.fsol = f

    def solToFunc2d(self):

        self.fsol = []
        self.fLims = []
        for i in range(len(self.elems)):
            xs, ys = self.elems[i].grid()
            self.fLims.append([np.array([xs[0], ys[0]]), np.array([xs[-1], ys[-1]])])
        self.fLims = np.array(self.fLims)
        f = lambda i, x, y: barycentric_interpolate(self.elems[i].grid()[1],
                                                    barycentric_interpolate(self.elems[i].grid()[0], self.sol[i].copy(), x).T, y)

        self.fsol = f
    def plot2d(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.gca(projection='3d')

        for i in range(len(self.elems)):
            flag = True
            grid = self.elems[i].grid()
            x1, x2 = approx.meshgrid(*grid)
            for j in range(self.mesh.n):
                if np.size(grid[j][np.isinf(grid[j])]) != 0:
                    flag = False
            if flag == True:
                #mehf = lambda x, y: np.exp(-a*((x)**2 + (y)**2))
                ax.plot_surface(x1, x2, self.sol[i][:, :])# - mehf(x1, x2))

        plt.show()

