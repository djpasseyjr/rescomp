import rescomp as rc
import numpy as np
import findiff as fd
from warnings import warn

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
    warn("The function `rescomp.accduration` is depreciated. Use `rescomp.valid_prediction_index` instead")
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

def nrmse(true, pred, axis=0):
    """ Normalized root mean squared error between two mxn arrays. 
        Parameters
        ----------
        true (ndarray): mxn array of the true values.
        pred (ndarray): mxn array of the predicted values
        axis (int): Can be 0 or 1. Decide which axis to compute the error
        Returns
        -------
        err (ndarray): If axis=0 returns length m array, if axis=1 returns length n array
    """
    sig = np.std(true, axis=axis, ddof=1) # Set ddof to match the R implementation of nrmse
    other_axis = (axis + 1 ) % 2 # Sends 0 -> 1 and 1 -> 0
    normalized_sq_err = (true - pred)**2 / sig**2
    if len(true.shape) == 1:
        err = np.mean(normalized_sq_err)**.5
    else:
        err = np.mean(normalized_sq_err, axis=other_axis)**.5
    return err

def valid_prediction_index(err, tol):
    """ First index i where err[i] > tol. 
        Parameters
        ----------
        err (ndarray): One dimensional array
        tol (float): Max allowable error. 
        Returns
        -------
        i (int): First index such that err[i] > tol
    """
    for i in range(len(err)):
        if err[i] > tol:
            return i
    return i

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
            The output of numpy.vstack((Utr, Uts)) will be an unbroken orbit
            from the given system.
        Parameters
        ----------
            system (str): A builtin rescomp system name ["rossler", "thomas", "lorenz"]
            duration (float): How long of an orbit
                Defaults to 10.0
            dt (float): Stepsize in time (For numerical integration)
                Defaults to 0.01
            trainper (float): Must be between 0 and 1. Percent of the orbit to place in training data
                Defaults to 0.5.
            trim (bool): If true, return an orbit of lenfth duration on the attractor. Otherwise,
                include the pre attractor orbit. (Defaults to True)
        Returns
        -------
            tr (ndarray): 1 dimensional array of time values corresponding to training orbit
            Utr (ndarray): 2 dimensional training orbit. Utr[i, :] is the state of the system at time tr[i]
            ts (ndarray): 1 dimensional array of time values corresponding to test orbit
            Uts (ndarray): 2 dimensional test orbit. Uts[i, :] is the state of the system at time ts[i]
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
