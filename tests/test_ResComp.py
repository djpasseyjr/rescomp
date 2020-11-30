import unittest
from rescomp import ResComp
import numpy as np
import itertools
import scipy as sp

PARAMCOMBOS  =  {
    "res_sz" : [100, 500],
    "activ_f" : [np.tanh, np.sin],
    "mean_degree" : [2.0, 3.0],
    "ridge_alpha" : [1e-4, 1.0],
    "spect_rad" : [.9, 2.0],
    "sparse_res" : [True, False],
    "sigma" : [0.1, 0.5],
    "uniform_weights" : [True, False],
    "gamma" : [1., 5.],
    "signal_dim" : [3, 1],
    "max_weight" : [10, 1],
    "min_weight" : [0, -1]
}

RES = {
    "res_sz": 50,
    "activ_f": np.tanh,
    "mean_degree": 2.0,
    "ridge_alpha": .0001,
    "spect_rad": 0.5,
    "gamma": 5.,
    "sigma": 1.5,
    "uniform_weights": True,
    "sparse_res": True,
    "signal_dim": 2,
    "map_initial" : "activ_f"
}

ADJCOMBOS = {
    "res_sz" : [100, 500],
    "rho" : [1.5, 2.0],
    "sparse" : [True, False]
}

def params_match(rcomp, keys, prms):
    for k, p in zip(keys, prms):
        rc_val = rcomp.__dict__[k]
        if  rc_val != p:
            if k in ["min_weight", "max_weight"]:
                # Reservoir edge weights are scaled to achieve
                # the correct spectral radius
                pass
            elif type(p) is not float:
                print("\n", k, rc_val, p, 1)
                return False
            elif np.abs(rc_val - p) > 0.1:
                print("\n", k, rc_val, p, 2)
                return False
    return True

def truejacobian(rcomp, r, u, trained=False):
    n = rcomp.res_sz
    if rcomp.activ_f == np.tanh:
        activ_f_prime = lambda x : 1 / np.cosh(x)**2
    if rcomp.activ_f == np.sin:
        activ_f_prime = lambda x : np.cos(x)
    if not trained:
        nonlin = activ_f_prime(rcomp.res @ r + rcomp.sigma*rcomp.W_in @ u)
        nonlin = np.reshape(nonlin, (-1,1))
        if rcomp.sparse_res:
            offdiag = rcomp.res.multiply(nonlin)
            J =  -1 * rcomp.gamma * (sp.sparse.eye(n) - offdiag)
        else:
            offdiag = rcomp.res * nonlin
            J =  -1 * rcomp.gamma * (np.eye(n) - offdiag)
    if trained:
        nonlin = activ_f_prime(rcomp.res @ r + rcomp.sigma*rcomp.W_in @ (rcomp.W_out @ r))
        nonlin = np.reshape(nonlin, (-1,1))
        if rcomp.sparse_res:
            offdiag = (rcomp.res.toarray() + rcomp.sigma * rcomp.W_in @ rcomp.W_out) * nonlin
            J =  -1 * rcomp.gamma * (sp.sparse.eye(n) - offdiag)
        else:
            offdiag = (rcomp.res + rcomp.sigma * rcomp.W_in @ rcomp.W_out)* nonlin
            J =  -1 * rcomp.gamma * (np.eye(n) - offdiag)
    return J

def jacobian_err(rcomp):
    r0 = np.random.rand(rcomp.res_sz)
    u0 = np.random.rand(rcomp.signal_dim)
    u = lambda x: u0
    # Untrained
    Jnum = rcomp.jacobian(0, r0, u, trained=False)
    J = truejacobian(rcomp, r0, u0, trained=False)
    untrainederr = np.max(np.abs(Jnum(r0) - J))
    # Trained
    rcomp.W_out = np.random.rand(*rcomp.W_out.shape)
    Jnum = rcomp.jacobian(0, r0, u, trained=True)
    J = truejacobian(rcomp, r0, u0, trained=True)
    trainederr = np.max(np.abs(Jnum(r0) - J))
    return np.max(untrainederr, trainederr)

def fixed_point_err(rcomp):
    u0 = np.random.rand(rcomp.signal_dim)
    u = lambda x: u0
    fixed_res_ode = lambda r: rcomp.res_ode(0, r, u)
    rstar_iter = sp.optimize.fsolve(fixed_res_ode, np.ones(rcomp.res_sz))
    rcomp.map_initial = "relax"
    rstar_relax = rcomp.initial_condition(u0)
    return np.max(np.abs(rstar_relax - rstar_iter))

def identity_adj(n=150, rho=1.0, sparse=False):
    A = sp.sparse.eye(n)
    if not sparse:
        A = A.toarray()
    A = rho * A
    datamembers = (n, np.tanh, 1.0, 1e-4, rho, sparse, 0.1, True, 1.0, 3, rho, rho)
    return A, datamembers

def nonuniform_adj(n=100, rho=1.0, sparse=False):
    A = sp.sparse.random(n, n, format='lil')
    A[0,0] = 1 # To ensure non-zero spectral radius
    spect_rad = np.max(np.abs(np.linalg.eigvals(A.toarray())))
    A = A / spect_rad
    if not sparse:
        A = A.toarray()
    A = rho * A
    maxw = float(rho / spect_rad)
    args = (n, np.tanh, n*0.01, 1e-4, rho, sparse, 0.1, False, 1.0, 3, maxw, 0.1)
    return A, args

def make_data():
    t = np.linspace(0, 20, 1000)
    U =  np.vstack((np.cos(t), -1 * np.sin(t))).T
    return t, U

def make_train_test_data():
    tr = np.linspace(0,20, 1000)
    ts = np.linspace(20,25, 500)
    signal = lambda x: np.vstack((np.cos(x), -1 * np.sin(x))).T
    return tr, ts, signal(tr), signal(ts)

# Test partition
def random_time_array(n, start=0):
    t = [start]
    def nextsample(t):
        t[0] += np.random.rand()
        return t[0]
    return [nextsample(t) for i in range(n)]

def uniform_time_array(n, start=0, end=500):
    return np.linspace(start, end, n)


class TestResComp(unittest.TestCase):
    def test_init_noargs(self):
        kwargs = dict()
        combos = itertools.product(*PARAMCOMBOS.values())
        keys = list(PARAMCOMBOS.keys())
        for c in combos:
            # For each combination of parameters, make a dictionary of kwargs
            for k, v in zip(keys, c):
                kwargs[k] = v
            # Initialize a reservoir computer
            rcomp = ResComp(**kwargs)
            # Check that the initialized rcomp has the right internal data
            assert params_match(rcomp, keys, c)

    def test_init_args(self):
        combos = itertools.product(*ADJCOMBOS.values())
        keys = list(PARAMCOMBOS.keys())
        for args in combos:
            I, prms = identity_adj(*args)
            rcomp = ResComp(I)
            assert params_match(rcomp, keys, prms)
            A, prms = nonuniform_adj(*args)
            rcomp = ResComp(A)
            assert params_match(rcomp, keys, prms)

    def test_drive(self):
        """ Drive the internal ode """
        t, U = make_data()
        rcomp = ResComp(**RES)
        r0 = rcomp.W_in @ U[0, :]
        out = rcomp.internal_state_response(t, U, r0)
        m, n = out.shape
        assert m == len(t) and n == rcomp.res_sz

    def test_fit(self):
        """ Make sure updates occur in the Tikhanov Factors"""
        rcomp = ResComp(**RES)
        t, U = make_data()
        rcomp.update_tikhanov_factors(t, U)
        assert not np.all(rcomp.Rhat == 0.0)
        assert not np.all(rcomp.Yhat == 0.0)

    def test_predict(self):
        """ Test that the reservoir can learn a simple signal"""
        rcomp = ResComp(**RES)
        t, U = make_data()
        rcomp.train(t, U)
        pre = rcomp.predict(t[500:], U[500, :])
        error = np.max(np.linalg.norm(pre - U[500:, :], ord=np.inf, axis=0))
        assert error < 0.5

    def test_predict_unseen(self):
        """ Predict on unseen data """
        rcomp = ResComp(**RES)
        tr, ts, Utr, Uts = make_train_test_data()
        rcomp.train(tr, Utr, window=10, overlap=0.9)
        pre = rcomp.predict(ts, Uts[0, :])
        error = np.mean(np.linalg.norm(pre - Uts, ord=2, axis=0)**2)**(1/2)
        assert error < 1.0

    def test_window(self):
        """ Make sure each partition is smaller than the given time window """
        rcomp = ResComp(**RES)
        for window in [.5, 3, 1001]:
            for timef in [random_time_array, uniform_time_array]:
                times = timef(1000)
                idxs = rcomp._partition(times, window, 0)
                for i,j in idxs:
                    sub = times[i:j]
                    assert sub[-1] - sub[0] <= window + 1e-12

    def test_overlap(self):
        """ Ensure that overlap is correct on average """
        rcomp = ResComp(**RES)
        for window in [30, 100]:
            for overlap in [.1, .9,]:
                T = 1000
                for times in [random_time_array(T), uniform_time_array(T)]:
                    idxs = rcomp._partition(times, window, overlap)
                    prev = None
                    over = 0.0
                    for i,j in idxs:
                        sub = times[i:j]
                        if prev is not None:
                            inters = set(sub).intersection(set(prev))
                            over += len(inters) / len(sub)
                        prev = sub
                    assert np.abs(over/len(idxs) - overlap) < .05

    def test_jacobian(self):
        kwargs = dict()
        combos = itertools.product(*PARAMCOMBOS.values())
        keys = list(PARAMCOMBOS.keys())
        for c in combos:
            # For each combination of parameters, make a dictionary of kwargs
            for k, v in zip(keys, c):
                kwargs[k] = v
            # Initialize a reservoir computer
            kwargs["res_sz"] = 10
            rcomp = ResComp(**kwargs)
            # Check that the initialized rcomp has the correct jacobian
            err = jacobian_err(rcomp)
            assert  err < 1e-8 , print("\n", kwargs, "Jacobian err:", err)

    # def test_fixed_point(self):
    #     kwargs = dict()
    #     combos = itertools.product(*PARAMCOMBOS.values())
    #     keys = list(PARAMCOMBOS.keys())
    #     for c in combos:
    #         # For each combination of parameters, make a dictionary of kwargs
    #         for k, v in zip(keys, c):
    #             kwargs[k] = v
    #         # Initialize a reservoir computer
    #         kwargs["res_sz"] = 10
    #         rcomp = ResComp(**kwargs)
    #         # Check that the initialized rcomp has the correct jacobian
    #         err = fixed_point_err(rcomp)
    #         assert  err < 1e-9 , print(kwargs, "Fixed point err:", err)


if __name__ == '__main__':
    unittest.main()
