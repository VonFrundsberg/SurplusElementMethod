from mathematics import approximate as approx
import numpy as np
from mathematics import spectral as spec
import matplotlib.pyplot as plt

x = spec.chebNodes(12, 0, 1)
xx, yy = approx.meshgrid(x, x)
evaluated_func = np.exp(-xx - yy) * (1 + np.sqrt(xx + yy))
func_TT = approx.simpleTTsvd(evaluated_func)
np.set_printoptions(precision=3, suppress=True)
# print(approx.toFullTensor(func_TT) - evaluated_func)
# print()
print(func_TT[1][:, :, 0].T + func_TT[0][0, :, :])