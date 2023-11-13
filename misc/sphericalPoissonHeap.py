import numpy as np
from GalerkinMethod.element.mainElementClass import element
import GalerkinMethod.element.elementUtils as operations
from mathematics import approximate as approx
from mathematics import integrate as integr
from mathematics import spectral as spec
import time as time
def solveSphericalPois(polyOrder, rhsF, integrPoints=350):

    elem = element(np.array([[0, np.inf], [0, np.pi], [0, 2*np.pi]]),
                   np.array(polyOrder, dtype=int), np.array([1, 0, 3]))

    rMatrixD = operations.integrateBilinearForm1_SameElement(elem, lambda x: x * x, integrPoints, 0)[:-1, :-1]
    rMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 0)[:-1, :-1]

    tMatrixD = operations.integrateBilinearForm1_SameElement(elem, lambda x: np.sin(x) ** 2, integrPoints, 1) + \
               operations.integrateBilinearForm2(elem, lambda x: np.sin(x) * np.cos(x), integrPoints, 1)

    tMatrixIr = operations.integrateBilinearForm0(elem, lambda x: np.sin(x)**2, integrPoints, 1)
    tMatrixIp = operations.integrateBilinearForm0(elem, lambda x: x*0 + 1.0, integrPoints, 1)

    pMatrixD = operations.integrateBilinearForm1_SameElement(elem, lambda x: x * 0 + 1.0, integrPoints, 2)
    pMatrixI = operations.integrateBilinearForm0(elem, lambda x: x * 0 + 1.0, integrPoints, 2)

    ttC = approx.kronSumtoTT_blockFormat([[None, -rMatrixD, rMatrixI],
                                         [tMatrixIr, -tMatrixD, tMatrixIp],
                                         [pMatrixI, -pMatrixD, None]])

    grid = elem.getGridList()
    w, idNodes = integr.reg_32_wn(-1, 1, integrPoints)
    grid[0] = elem[0].map(idNodes)
    grid = approx.meshgrid(*grid)

    fx = rhsF(grid)
    fx = np.nan_to_num(fx, 0)

    ttFx = approx.simpleTTsvd(fx, tol=1e-6, R_MAX=400)
    # for it in ttFx:
    #     print(it.shape)
    # time.sleep(500)
    # grid0 = idNodes
    # core0 = ttFx[0]
    # core0 = np.transpose(core0, [1, 0, 2])
    # core0 = spec.barycentricChebInterpolate(core0, grid0, a=-1, b=1, extrapolation=0, axis=0)
    # core0 = np.transpose(core0, [1, 0, 2])
    # ttFx[0] = core0

    grid1 = idNodes
    core1 = ttFx[1]
    core1 = np.transpose(core1, [1, 0, 2])
    core1 = spec.barycentricChebInterpolate(core1, grid1, a=-1, b=1, extrapolation=0, axis=0)
    core1 = np.transpose(core1, [1, 0, 2])
    ttFx[1] = core1
    # print(core1.shape)
    grid2 = idNodes
    core2 = ttFx[2]
    core2= np.transpose(core2, [1, 0, 2])
    core2 = np.einsum('ij, jnk -> ink', elem[2].eval(grid2), core2)

    core2 = np.transpose(core2, [1, 0, 2])
    ttFx[2] = core2

    # print('cores of f tt decomposition have the following shapes: ')
    # for i in ttFx:
    #     print(i.shape)
    # time.sleep(500)
    ttR = ttFx[0].copy()
    ttR = np.reshape(ttR, [ttR.shape[1], ttR.shape[2]])
    integral = operations.integrateFunctional(elem, ttR, integrPoints, 0, True, precalc=True)
    integral = integral[np.newaxis, :-1, :]
    ttFx[0] = integral

    ttT = ttFx[1].copy()
    preshape = ttT.shape
    ttT = np.transpose(ttT, [1, 0, 2])
    ttT = np.reshape(ttT, [preshape[1], preshape[0]*preshape[2]])
    integral = operations.integrateFunctional(elem, ttT, integrPoints, 1, True, precalc=True)
    integral = integral[:, :, np.newaxis]
    integral = np.reshape(integral, [elem[1].approxOrder, preshape[0], preshape[2]])
    integral = np.transpose(integral, [1, 0, 2])
    ttFx[1] = integral
    ttP = ttFx[2].copy()
    preshape = ttP.shape
    ttP = np.transpose(ttP, [1, 0, 2])
    ttP = np.reshape(ttP, [preshape[1], preshape[0] * preshape[2]])
    integral = operations.integrateFunctional(elem, ttP, integrPoints, 2, True, precalc=True)
    integral = integral[:, :, np.newaxis]
    integral = np.reshape(integral, [elem[2].approxOrder, preshape[0], preshape[2]])
    integral = np.transpose(integral, [1, 0, 2])
    ttFx[2] = integral
    # r, t, p = elem.getGridList()
    # rr, tt, pp = approx.meshgrid(r[:-1], t, p)
    # grid = approx.meshgrid(r[:-1], t, p)
    # fx = -np.sqrt(rr**2 + tt**2 + pp**2)
    # ttFx = approx.simpleTTsvd(fx)
    # ttfx = approx.alterLeastSquares(ttC, ttFx, np.ones([elem.dim, 2]))
    # print("difference between tensors is: ", np.max(np.abs(approx.toFullTensor(ttFx) - fx)))
    fx = (approx.toFullTensor(ttFx)).flatten()
    # anothertt = approx.simpleTTsvd(np.reshape(C.dot(fx.flatten()), elem.approxOrder - [1, 0, 0]))
    # for it in anothertt:
    #     print("another tt shape", it.shape)
    # print(np.max(np.abs(C.dot(fx.flatten()) - approx.toFullTensor(ttfx).flatten())))
    # time.sleep(500)
    t = time.time()
    # print("everything is calculated, starting system solving")
    # ttSol = sp_lin.solve(C, fx)
    # ttSol = sp_lin.sol
    # print("sparse solver done in ", time.time() - t)
    # print("solving with TT als algorithm")
    # t = time.time()
    tf3tensor = ttpy.TensorTrain(ttC)
    ttpy.sol
    print(tf3tensor)
    print('done')
    time.sleep(500)
    ttSol = approx.alterLeastSquares(ttC, ttFx, 7).matricize()
    print(ttSol.shape)
    T = time.time() - t
    # print("ALS solver done in ", time.time() - t)
    # print("difference is ", np.max(np.abs(ttSol - sol)))
    # print("max is: ", np.max(np.abs(sol)))
    # sol = np.reshape(sol, elem.approxOrder - [1, 0, 0])
    # ttFx = approx.simpleTTsvd(sol, tol=1e-6)
    # print("solution tt ranks")
    # for i in ttFx:
    #     print(i.shape)
    r, t, p = elem.getGridList()
    # print(np.cos(2*t))
    rr, tt, pp = approx.meshgrid(r[:-1], t, p)
    grid = approx.meshgrid(r[:-1], t, p)
    fx = -(np.exp(-rr) * (np.sin(pp)) * np.cos(2 * tt) + np.exp(-2*rr) * (np.cos(2*pp)) * np.cos(4 * tt))

    print(T, np.max(np.abs(ttSol + fx.flatten())))

def solveSphericalPoisForPlot(iStart, iEnd, iStep):

    for i in range(iStart, iEnd, iStep):
        solveSphericalPois(np.array([i, i, min(i, 14)]), lambda x: np.exp(-x[0])*(np.sin(x[2])*(np.cos(2*x[1])*(-1.0 + (-4.0 + (-2.0 + x[0])*x[0])*(np.sin(x[1])**2)) - (np.sin(2*x[1])**2)) +
                                          np.exp(-x[0])*4*np.cos(2*x[2])*(np.cos(4*x[1])*(-1.0 + (-4.0 + (-1.0 + x[0])*x[0])*(np.sin(x[1])**2)) - \
                                                                    np.cos(x[1])*np.sin(x[1])*np.sin(4*x[1])))
                       , integrPoints=500)

solveSphericalPoisForPlot(4, 40, 2)