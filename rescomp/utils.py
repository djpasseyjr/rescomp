import rescomp as rc
import numpy as np
import findiff as fd

def L2(x, axis=0):
    return np.sum(x**2, axis=axis)**0.5
def Linf(x, axis=0):
    return np.max(np.abs(x), axis=axis)

def relerr(true, pre, order=2, axis=0):
    if order == 2:
        norm = L2
    if order == "inf":
        norm = Linf
    return norm(true - pre, axis=axis) / norm(true, axis=axis)

def accduration(true, pre, tol=0.2, order="inf", axis=0):
    n = pre.shape[axis]
    for i in range(n):
        if axis == 0:
            t = true[i, :]
            p = pre[i, :]
        if axis == 1:
            t = true[:, i]
            p = pre[:, i]
        if relerr(t, p, order=order, axis=0) > tol:
            return i
    return n - 1

def system_fit_error(t, U, system, order="inf"):
    dt = np.mean(np.diff(t))
    ddt = fd.FinDiff(0, dt, acc=6)
    df = rc.SYSTEMS[system]['df']
    err = ddt(U) - df(t, U)
    if order == "inf":
        return np.max(np.abs(err))
    if order == 2:
        return np.mean(np.sum(err**2, axis=1)**0.5)

def train_test_orbit(system, duration=10, dt=0.01, trainper=0.5, trim=True):
    """ Returns a time scale and orbit of length `duration` split into two pieces:
            tr, Utr, ts, Uts
        where `tr` contains `trainper` percent of the total orbit.
        Therefore the output of numpy.vstack((Utr, Uts)) is an unbroken orbit
        from the given system.
        By default trims off the portion of the orbit that is not on the attractor.
    """
    t, U = rc.orbit(system, duration=duration, dt=dt, trim=trim)

    N = len(t)
    mid = int(N * trainper)
    tr, Utr = t[:mid], U[:mid, :]
    ts, Uts = t[mid:], U[mid:, :]
    return tr, Utr, ts, Uts


def lyapunov(t, x, xdelta, delta0):
    """ Estimate the largest lyapunov exponent according to the algorithm in
                Michael T. Rosenstein, James J. Collins, Carlo J. De Luca,
                A practical method for calculating largest Lyapunov exponents from small data sets,
                Physica D: Nonlinear Phenomena,
                Volume 65, Issues 1–2,
                1993.
        Parameters
        ----------
        t (array): n time values
        x (array): nxd matrix where d is the dimension of the system. The ith row
                  of x is equal to the state of the system at time t[i]
        xd (array): nxd matrix of system states when x(0) is perturbed by delta
        delta (array): Array of length d. Perturbation to initial condition

        Returns
        -------
        lam (float): Estimate of the largest lypunov exponent
    """
    delta = x - xdelta
    deltanorm = np.sum(delta**2, axis=1)**0.5
    delta0norm = np.sum(delta**2)**0.5
    logdelta = np.log(deltanorm/delta0norm)
    one = np.ones(len(t))
    A = np.vstack((one, t)).T
    x, res, rank, sigma = np.linalg.lstsq(A, logdelta)
    lam = x[1]
    return lam
