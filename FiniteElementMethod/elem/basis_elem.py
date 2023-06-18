import numpy as np
from misc import spectral as sp

class b_elem():
    def __init__(self, I=[-1.0, 1.0], n=1, bc=[], Id=None, infMap="linear"):
        self.I = np.array(I)
        self.n = n
        self.bc = bc
        if Id == None:
            self.Id = np.eye(n)
        else:
            self.Id = Id
        self.D = sp.chebDiff(self.n).dot(self.rp_eval())
        # print(infMap)
        if infMap == "linear":
            if I[0] == -np.inf:
                self.map = lambda x: -((1.0 - x) / (1.0 + x) - I[1])
                self.imap = lambda x: (x + 1.0 - I[1])/(-x + I[1] + 1.0)

                self.dmap = lambda x: (x + 1) ** 2 / 2
                self.idmap = lambda x: 2/(x + 1)**2
                return

            if I[1] == np.inf:
                self.map = lambda x: ((1.0 + x) / (1.0 - x) + I[0])
                self.imap = lambda x: (-x + I[0] + 1.0) / (-x + I[0] - 1.0)

                self.dmap = lambda x: (x - 1) ** 2 / 2
                self.idmap = lambda x: 2/(x - 1) ** 2
                return
        elif infMap == "exponential":
            if I[0] == -np.inf:
                self.map = lambda x: -((1.0 - x) / (1.0 + x) - I[1])
                self.imap = lambda x: (x + 1.0 - I[1])/(-x + I[1] + 1.0)

                self.dmap = lambda x: (x + 1) ** 2 / 2
                self.idmap = lambda x: 2/(x + 1)**2
                return

            if I[1] == np.inf:
                self.map = lambda x: np.log((1.0 + x) / (1.0 - x) + 1) + I[0]
                self.imap = lambda x: np.exp(I[0] - x)*(np.exp(x - I[0]) - 2.0)

                self.dmap = lambda x: (1.0 - x)
                self.idmap = lambda x: 1/(1.0 - x)
                return

        a = I[0]; b = I[1]
        q = (b - a) / 2.0; p = (b + a) / 2.0
        self.map = lambda x: (q * x + p)
        self.imap = lambda x: (x - p)/q

        self.dmap = lambda x: 1.0 / (q + x * 0)
        self.idmap = lambda x: q + 0

    def rp_eval(self):
        Id = self.Id
        for it in self.bc:
            Id[it[0], it[0]] = it[1]
        return Id

    def p_eval(self, x): ## return shape: (*x.shape, deg of element)
        x = np.atleast_1d(x)
        P = self.rp_eval()
        if x.size == 1:
            if x[0] == self.I[0]:
                return np.array([P[0, :]])
            if x[0] == self.I[-1]:
                return np.array([P[-1, :]])
            # print('oops')
        # xs = np.array(self.imap(x), dtype=np.float)
        res = sp.bary(P, x)
        return res

    def xs(self):
        x = sp.chebNodes(self.n)
        x = self.map(x)
        return x

    def new_xs(self):
        x = sp.chebNodes(self.n)
        mx = self.imap

        return x, mx

    def dp_eval(self, x): ## return shape: (*x.shape, deg of element)
        x = np.atleast_1d(x)
        P = self.D
        if x.size == 1:
            if x[0] == self.I[0]:
                return self.dmap(-1)*np.array([P[0, :]])
            if x[0] == self.I[-1]:
                return self.dmap(1)*np.array([P[-1, :]])
        # xs = np.array(self.imap(x), dtype=np.float)
        res = sp.bary(f=P, x=x)*np.reshape(self.dmap(x), (*x.shape, 1))
        return res
    def gen_func(self):
        return lambda x: self.p_eval(x)
    def gen_funcd(self):
        return lambda x: self.dp_eval(x)

    def __call__(self, x):
            return self.p_eval(x)
    def d(self, x):
            return self.dp_eval(x)