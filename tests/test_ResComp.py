import unittest
from rescomp import *
import numpy as np
import itertools
import scipy as sp

PARAMCOMBOS  =  {
    "res_sz" : [100, 500],
    "activ_f" : [np.tanh, np.arctan],
    "mean_degree" : [2.0, 3.0],
    "ridge_alpha" : [1e-4, 1.0],
    "spect_rad" : [.9, 2.0],
    "sparse_res" : [True, False],
    "sigma" : [0.1, 0.5],
    "uniform_weights" : [True, False],
    "gamma" : [1., 5.],
    "signal_dim" : [3, 1],
    "max_weight" : [0, 1],
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
    "signal_dim": 2
}

ADJCOMBOS = {
    "res_sz" : [100, 500],
    "rho" : [1.5, 2.0],
    "sparse" : [True, False]
}

def params_match(rc, keys, prms):
    for k, p in zip(keys, prms):
        rc_val = rc.__dict__[k]
        if  rc_val != p:
            if type(p) is not float:
                print(k, rc_val, p, 1)
                return False
            elif np.abs(rc_val - p) > 0.1:
                print(k, rc_val, p, 2)
                return False
    return True

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
    datamembers = (n, np.tanh, n*0.01, 1e-4, rho, sparse, 0.1, False, 1.0, 3, maxw, 0.1)
    return A, datamembers

def make_data():
    t = np.linspace(0, 20, 1000)
    U =  np.vstack((np.cos(t), -1 * np.sin(t))).T
    return t, U

def make_train_test_data():
    tr = np.linspace(0,10, 500)
    ts = np.linspace(10,15, 250)
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
        params = dict()
        combos = itertools.product(*PARAMCOMBOS.values())
        keys = list(PARAMCOMBOS.keys())
        for c in combos:
            for k, v in zip(keys, c):
                params[k] = v
        # Initialize reservoir
        rc = ResComp(**params)
        # Check that data members are correct
        for c in combos:
            assert params_match(rc, keys, c)

    def test_init_args(self):
        combos = itertools.product(*ADJCOMBOS.values())
        keys = list(PARAMCOMBOS.keys())
        for args in combos:
            I, prms = identity_adj(*args)
            rc = ResComp(I)
            assert params_match(rc, keys, prms)
            A, prms = nonuniform_adj(*args)
            rc = ResComp(A)
            assert params_match(rc, keys, prms)

    def test_drive(self):
        """ Drive the internal ode """
        t, U = make_data()
        rc = ResComp(**RES)
        r0 = rc.W_in @ U[0, :]
        out = rc.internal_state_response(t, U, r0)
        m, n = out.shape
        assert m == len(t) and n == rc.res_sz

    def test_fit(self):
        """ Make sure updates occur in the Tikhanov Factors"""
        rc = ResComp(**RES)
        t, U = make_data()
        rc.update_tikhanov_factors(t, U)
        assert not np.all(rc.Rhat == 0.0)
        assert not np.all(rc.Yhat == 0.0)

    def test_predict(self):
        """ Test that the reservoir can learn a simple signal"""
        rc = ResComp(**RES)
        t, U = make_data()
        rc.train(t, U)
        pre = rc.predict(t[500:], U[500, :])
        error = np.max(np.linalg.norm(pre - U[500:, :], ord=np.inf, axis=0))
        assert error < 0.5

    def test_predict_unseen(self):
        """ Predict on unseen data """
        rc = ResComp(**RES)
        tr, ts, Utr, Uts = make_train_test_data()
        rc.train(tr, Utr)
        pre = rc.predict(ts, Uts[0, :])
        error = np.mean(np.linalg.norm(pre - Uts, ord=2, axis=0)**2)**(1/2)
        assert error < 1.0

    def test_window(self):
        """ Make sure each partition is smaller than the given time window """
        rc = ResComp(**RES)
        for window in [.5, 3, 1001]:
            for timef in [random_time_array, uniform_time_array]:
                times = timef(1000)
                idxs = rc._partition(times, window, 0)
                for i,j in idxs:
                    sub = times[i:j]
                    assert sub[-1] - sub[0] <= window + 1e-12

    def test_overlap(self):
        """ Ensure that overlap is correct on average """
        rc = ResComp(**RES)
        for window in [30, 100]:
            for overlap in [.1, .9,]:
                T = 1000
                for times in [random_time_array(T), uniform_time_array(T)]:
                    idxs = rc._partition(times, window, overlap)
                    prev = None
                    over = 0.0
                    for i,j in idxs:
                        sub = times[i:j]
                        if prev is not None:
                            inters = set(sub).intersection(set(prev))
                            over += len(inters) / len(sub)
                        prev = sub
                    assert np.abs(over/len(idxs) - overlap) < .05

if __name__ == '__main__':
    unittest.main()
