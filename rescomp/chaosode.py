import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def unpack(X, cols=True):
    """ Splits 2d numpy arrays into tuples 
        Parameters
        ----------
        X (ndarray): 2d numpy array
        cols (bool): If True, split the array into column vectors, if false split it into row vectors
        
        Returns
        -------
        unpack (tuple): A tuple of the rows of X or a tuple of the columns of X. 
            If X is not a 2d numpy array, unpack=X.
    """
    if type(X) is np.ndarray:
        if len(X.shape) > 1:
            m, n = X.shape
            if cols:
                unpack = tuple([np.reshape(X[:,i], (m, 1)) for i in range(n)])
            else:
                unpack = tuple([X[i, ] for i in range(m)])
            return unpack
    return X


def lorenz(t, X, sigma=10., beta=8./3, rho=28.0):
    """Compute the time-derivative of a Lorenz system."""
    (x, y, z) = unpack(X)
    return np.hstack((sigma * (y - x), x * (rho - z) - y, x * y - beta * z))

def rossler(t, X):
    """ Compute the time derivative of the Rossler System """
    (x, y, z) = unpack(X)
    return np.hstack((-1 * (y + z), x + y / 5, 0.2 + z * (x - 5.7)))

def thomas(t, X, b=0.1998):
    """ Compute time derivative of Thomas' cyclically symmetric system
            Thomas, R. [1999] “Deterministic chaos seen in terms of feedback
            circuits: Analysis, synthesis, ‘labyrinth chaos’,” Int. J.
            of Bifurcation and Chaos 9, 1889–1905.
    """
    (x, y, z) = unpack(X)
    return np.hstack((np.sin(y) - b * x, np.sin(z) - b * y, np.sin(x) - b * z))

SYSTEMS = {
    "lorenz" : {
        "domain_scale" : [20, 20, 20],
        "domain_shift" : [-10, -10, 10],
        "signal_dim" : 3,
        "time_to_attractor" : 40.0,
        "df" : lorenz,
        "rcomp_params" : {
                        "res_sz" : 1000,
                        "activ_f" : lambda x: 1/(1 + np.exp(-1*x)),
                        "gamma" : 5.632587,
                        "mean_degree" : 0.21,
                        "ridge_alpha" : 2e-7,
                        "sigma" : 0.078,
                        "spect_rad" : 14.6
        }
    },
    "rossler" : {
        "domain_scale" : [10, 10, 0],
        "domain_shift" : [-5, -5, 0],
        "signal_dim" : 3,
        "time_to_attractor" : 40.0,
        "df" : rossler,
        "rcomp_params" : {
                        "res_sz" : 1000,
                        "activ_f" : lambda x: 1/(1 + np.exp(-1*x)),
                        "gamma" : 19.1,
                        "mean_degree" : 2.0,
                        "ridge_alpha" : 6e-7,
                        "sigma" : 0.063,
                        "spect_rad" : 8.472
        }
    },
    "thomas" : {
        "domain_scale" : [-2, -2, -2],
        "domain_shift" : [-2, -2, -2],
        "signal_dim" : 3,
        "time_to_attractor" : 100.0,
        "df" : thomas,
        "rcomp_params" : {
                        "res_sz" : 1000,
                        "activ_f" : lambda x: 1/(1 + np.exp(-1*x)),
                        "gamma" : 12.6,
                        "mean_degree" : 2.2,
                        "ridge_alpha" : 5e-4,
                        "sigma" : 1.5,
                        "spect_rad" : 12.0
        }
    }
}

def random_initial(system):
    a = SYSTEMS[system]["domain_scale"]
    b = SYSTEMS[system]["domain_shift"]
    dim = SYSTEMS[system]["signal_dim"]
    u0 = a * np.random.rand(dim) + b
    return u0

def orbit(system, initial=None, duration=10, dt=0.01, trim=False):
    """ Returns the orbit of a given system.
       
        Parameters
        ----------
        system (str): A supported dynamical system from ["lorenz", "thomas", "rossler"]
        initial (ndarray): An initial condition for the system. Defaults to a random choice.
        duration (float): Time duration of the orbit (default duration=10 means 10 seconds)
        dt (float): Timestep size. Default dt=0.01
        trim (bool): Option to trim off transient portion of the orbit (To ensure the orbit 
            is on the chaotic attractor for the full duration.)
            
        Returns
        -------
        U (ndarray): mxn numpy array where m is the number of timesteps (duration x dt) and n is 
            dimension of the system (probably 3).
    """
    transient_timesteps = 0
    time_to_attractor = 0
    if initial is None:
        initial = random_initial(system)
    if trim:
        time_to_attractor = SYSTEMS[system]["time_to_attractor"]
        transient_timesteps = int(time_to_attractor / dt)
    timesteps = int(duration / dt) + 1
    # Make enough timesteps so that the transients can be trimmed leaving a full duration orbit
    t = np.linspace(0, time_to_attractor + duration, transient_timesteps + timesteps)
    # Locate the correct derivative function
    df = SYSTEMS[system]["df"]
    U = integrate.odeint(df, initial, t, tfirst=True)
    # Trim off transient states
    U, t = U[transient_timesteps: , :], t[transient_timesteps:]
    return t, U

def plot3d(U, color=None):
    if color is None:
        color = np.random.rand(3)
    ax = plt.axes(projection='3d')
    x,y,z = U[:,0], U[:, 1], U[:, 2]
    ax.plot3D(x, y, z, c=color,alpha=.5)
    ax.set_title = "3D Orbit"
    plt.show()
    return color

def plot2d(t, U, color=None):
    if color is None:
        color = np.random.rand(U.shape[1], 3)
    for i, c in enumerate(color):
        plt.plot(t, U[:, i], c=c)
    plt.show()
    return color
