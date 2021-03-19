from .ResComp import ResComp
import numpy as np
from scipy.interpolate import CubicSpline
from scipy import integrate

class DrivenResComp(ResComp):
    """ Reservoir Computer that learns a response to a input signal
    """
    def __init__(self, *args, drive_dim=1, delta=.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.drive_dim = drive_dim
        self.delta = delta
        self.W_drive = 2*np.random.rand(self.res_sz, self.drive_dim) - 1.0

    def res_f(self, t, r, u, d):
        """ ODE to drive the reservoir node states with u(t) and a input signal d(t)"""
        transform_train = self.sigma * self.W_in @ u(t)
        transform_drive = self.delta * self.W_drive @ d(t)
        return self.gamma * (-1 * r + self.activ_f(self.res @ r + transform_train + transform_drive))

    def res_pred_f(self, t, r, d):
        """ Reservoir prediction ode. Assumes precomputed W_out. Accepts an input signal d(t) """
        recurrence = self.sigma * self.W_in @ (self.W_out @ r)
        transform_drive =  self.delta * self.W_drive @ d(t)
        return self.gamma*(-1*r + self.activ_f(self.res @ r + recurrence + transform_drive))

    def initial_condition(self, u0, d0):
        """ Function to map external system initial conditions to reservoir initial conditions """
        # u = lambda x: u0
        # d = lambda x: d0
        # fixed_res_f = lambda r: self.res_f(0, r, u, d)
        # r0 = optimize.fsolve(fixed_res_f, np.random.rand(self.res_sz))
        r0 = self.activ_f(self.W_in @ u0)
        return r0

    def train(self, t, U, D, window=None, overlap=0):
        """ Train the reservoir computer so that it can replicate the data in U.

            Paramters
            ---------
            t (1-d array or list of 1-d arrays): Array of m equally spaced time values corresponding to signal U.
            U (array or list of arrays): Input signal array (m x self.signal_dim) where the ith row corresponds to the
                training signal value at time t[i]
            D (array): For each i, D[i, :] produces the state of the input signal
                at time t[i]
            window (float): If window is not `None` the reservoir computer will subdivide the input signal
                into blocks where each block corresponds to `window` seconds of time.
                Defaults to None
            overlap (float): Must be less than one and greater or equal to zero. If greater than zero, this
                will cause subdivided input signal blocks to overlap. The `overlap` variable specifies the
                percent that each signal window overlaps the previous signal window
                Defaults to 0.0
        """
        if isinstance(t, list) and isinstance(U, list) and isinstance(D, list):
            for time, signal, input in zip(t, U, D):
                idxs = self._partition(time, window, overlap=overlap)
                for start, end in idxs:
                    ti = time[start:end]
                    Ui = signal[start:end, :]
                    Di = input[start:end, :]
                    self.update_tikhanov_factors(ti, Ui, Di)
        else:
            idxs = self._partition(t, window, overlap=overlap)
            for start, end in idxs:
                ti = t[start:end]
                Ui = U[start:end, :]
                Di = D[start:end, :]
                self.update_tikhanov_factors(ti, Ui, Di)
        self.W_out = self.solve_wout()
        self.is_trained = True

    def update_tikhanov_factors(self, t, U, D):
        """ Drive the reservoir with the u and collect state information into
            self.Rhat and self.Yhat
            Parameters
            t (1 dim array): array of time values
            U (array): for each i, U[i, :] produces the state of the training signal
                at time t[i]
            D (array): For each i, D[i, :] produces the state of the input signal
                at time t[i]
        """
        # The i + batchsize + 1 ending adds one timestep of overlap to provide
        # the initial condition for the next batch. Overlap is removed after
        # the internal states are generated
        idxs = [(i, i + self.batchsize + 1) for i in range(0, len(t), self.batchsize)]
        # Set initial condition for reservoir nodes
        r0 = self.initial_condition(U[0, :], D[0, :])
        for start, end in idxs:
            ti = t[start:end]
            Ui = U[start:end, :]
            Di = D[start:end, :]
            states = self.internal_state_response(ti, Ui, Di, r0)
            # Get next initial condition and trim overlap
            states, r0 = states[:-1, :], states[-1, :]
            # Update Rhat and Yhat
            self.Rhat += states.T @ states
            self.Yhat += Ui[:-1, :].T @ states
        self.r0 = r0

    def internal_state_response(self, t, U, D, r0):
        """ Drive the reservoir node states with the training signal U and input signal D
            Parameters
            ----------
            t (1 dim array): array of time values
            U (array): for each i, U[i, :] produces the state of the training signal
                at time t[i]
            r0 (array): Initial condition of reservoir nodes
            D (array): For each i, D[i, :] produces the state of the input signal
                at time t[i]
            Returns
            -------
            states (array): A (len(t) x self.res_sz) array where states[i, :] corresponds
                to the reservoir node states at time t[i]

        """
        u = CubicSpline(t, U)
        d = CubicSpline(t, D)
        states = integrate.odeint(self.res_f, r0, t, tfirst=True, args=(u,d))
        return states

    def predict(self, t, D, u0=None, r0=None, return_states=False):
        """ Drive the reservoir node states with the training signal U and input signal D

            Parameters
            ----------
            t (1 dim array): array of time values
            D (array): for each i, D[i, :] produces the state of the input signal
                at time t[i]
            u0 (array): Initial condition of the learned system
            r0 (array): Alternately supply initial condition for reservoir nodes

            Returns
            -------
            pred (array): A (len(t) x self.res_sz) array where states[i, :] corresponds
                to the reservoir node states at time t[i]
            states (array): The state of the reservoir nodes for each time in t.
                Optional. Returned if `return_states=True`.
        """
        # Determine initial condition
        if (u0 is not None):
            r0 = self.initial_condition(u0, D[0, :])
        elif r0 is None :
            r0 = self.r0
        if not self.is_trained:
            raise Exception("Reservoir is untrained")
        d = CubicSpline(t, D)
        states = integrate.odeint(self.res_pred_f, r0, t, tfirst=True, args=(d,))
        pred = self.W_out @ states.T
        # Return internal states as well as predicition or not
        if return_states:
            return pred.T, states
        return pred.T
