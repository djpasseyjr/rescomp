
import numpy as np
from findiff import FinDiff

def random_slice(*args, axis=0):
    """ Take a random slice of an arbitrary number of arrays from the same index
        Parameters:
            As (ndarrays): Arbitrary number of arrays with the same size along the given axis
            slicesize (int): Size of random slice must be smaller than the size of the
                arrays along given axis
        Keyword Parameters:
            axis (int): Axis to slice. Can be 0 or 1.
        Returns
            slices (tuple): A tuple of slices from each array
    """
    As, slicesize = args[:-1], args[-1]
    start = np.random.randint(0, high=len(As[0]) - slicesize + 1)
    end = start + slicesize
    if axis == 0:
        slices = (A[start:end] for A in As)
    if axis == 1:
        slices = (A[:, start:end] for A in As)
    return slices
    
class System:
    """
    Abstract class for defining a system for a reservoir computer to train and predict on.
    Requires implementing get_train_test_data(), get_random_test(), and optionally df_eq()
    """
    def __init__(self, name, train_time, test_time, dt, signal_dim, drive_dim=0, is_diffeq=False, is_driven=False):
        self.name=name
        self.train_time=train_time
        self.test_time=test_time
        self.dt=dt
        self.signal_dim=signal_dim
        self.drive_dim=drive_dim
        self.is_diffeq=is_diffeq
        self.is_driven=is_driven
    
    def get_train_test_data(self, cont_test=True):
        """
        Returns training and test data for use by a reservoir computer.
        
        Arguments:
            cont_test (bool): True for continued test, False for random test
        
        Returns:
            If not driven:
                tr, Utr, ts, Uts
            If driven:
                tr, (Utr, Dtr), (ts, Dts), Uts
        Where:
            tr, ts ((n,) ndarray): training/test time values
            Utr, Uts ((n,m) ndarray): training/test system state values
            Dtr, Dts ((n,l) ndarray): training/test system driving values
        """
        raise NotImplemented("this function has not been implemented")
    
    def get_random_test(self):
        """
        Returns test data from an arbitrary initial condition.
        
        Returns:
            If not driven:
                ts, Uts
            If driven:
                (ts, Dts), Uts
        Where:
            ts ((n,) ndarray): training/test time values
            Uts ((n,m) ndarray): training/test system state values
            Dts ((n,l) ndarray): training/test system driving values
        """
        raise NotImplemented("this function has not been implemented")
    
    def df_eq(self, t, U, D=None):
        """
        The differential equation governing the system, if applicable.
        Only used if self.is_diffeq is True.
        
        Parameters:
            t: 1-dimensional array, the timesteps of the system
            U: 2-dimensional array, describing the system state at either one point or over a time series.
            D: 2-dimensional array, describing the system's drive state at either one point or over a time series. Only ever passed if self.is_driven is True.
                U[t,:] is the system's state at time t, and likewise for D.
        """
        raise NotImplemented("this function has not been implemented")
        
    ##############################################################################
    
    def system_fit_error(self, t, U, order="inf"):
        """
        Computes the system fit error if the system has a corresponding differential equation
        Based on rescomp.system_fit_error().
        Parameters:
            t (ndarray): time steps
            U (ndarray): system values, whether actual or estimated
            order ('inf' or float): the order of norm to use
        """
        if not self.is_diffeq:
            raise RuntimeError("cannot compute system fit error for system without a differential equation")
        if self.is_driven:
            t, D = t
        
        dt = np.mean(np.diff(t))
        ddt = FinDiff(0, dt, acc=6)
        
        if self.is_driven:
            err = ddt(U) - self.df(t, U, D)
        else:
            err = ddt(U) - self.df(t, U)
        
        if order == "inf":
            return np.max(np.abs(err))
        else:
            return np.mean(np.linalg.norm(err,axis=1,ord=order))
            
    def df(self, t, U, D=None):
        """
        The differential equation governing the system, if applicable.
        Only used if self.is_diffeq is True
        
        Parameters:
            t: float or 1-d array: the timesteps to apply the system differential equation at. Generally, this is unused internally.
            U, D: 1- or 2-dimensional arrays, describing the system state at either one point or over a time series. D is not used if the system is not driven.
        """
        if not self.is_diffeq:
            raise RuntimeError("this system does not have an associated differential equation")
        if D is not None and not self.is_driven:
            raise ValueError("cannot specify drive state for non-driven system")
        if D is None and self.is_driven:
            raise ValueError("must specify drive state for driven system")
        
        shape = U.shape
        t = np.atleast_1d(t)
        U = np.atleast_2d(U)
        if D is not None:
            D = np.atleast_2d(D)
        return self.df_eq(U,D).reshape(shape)
        
        
        
        
        
        
        
    