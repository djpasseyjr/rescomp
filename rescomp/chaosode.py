import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def lorenz(t, X, sigma=10., beta=8./3, rho=28.0):
    """Compute the time-derivative of a Lorenz system."""
    (x, y, z) = X
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

def rossler(t, X):
    """ Compute the time derivative of the Rossler System """
    (x, y, z) = X
    return [-1 * (y + z), x + y / 5, 0.2 + z * (x - 5.7)]

def thomas(t, X, b=0.1998):
    """ Compute time derivative of Thomas' cyclically symmetric system
            Thomas, R. [1999] “Deterministic chaos seen in terms of feedback
            circuits: Analysis, synthesis, ‘labyrinth chaos’,” Int. J.
            of Bifurcation and Chaos 9, 1889–1905.
    """
    (x, y, z) = X
    return [np.sin(y) - b * x, np.sin(z) - b * y, np.sin(x) - b * z]

SYSTEMS = {
    "lorenz" : {
        "domain_scale" : [20, 20, 20],
        "domain_shift" : [-10, -10, 10],
        "signal_dim" : 3,
        "time_to_attractor" : 40.0,
        "df" : lorenz
    },
    "rossler" : {
        "domain_scale" : [10, 10, 0],
        "domain_shift" : [-5, -5, 0],
        "signal_dim" : 3,
        "time_to_attractor" : 40.0,
        "df" : rossler
    },
    "thomas" : {
        "domain_scale" : [-2, -2, -2],
        "domain_shift" : [-2, -2, -2],
        "signal_dim" : 3,
        "time_to_attractor" : 100.0,
        "df" : thomas
    }
}

def random_initial(system):
    a = SYSTEMS[system]["domain_scale"]
    b = SYSTEMS[system]["domain_shift"]
    dim = SYSTEMS[system]["signal_dim"]
    u0 = a * np.random.rand(dim) + b
    return u0

def orbit(system, initial=None, duration=10, dt=0.01, trim=False):
    transient_timesteps = 0
    time_to_attractor = 0
    if initial is None:
        initial = random_initial(system)
    if trim:
        time_to_attractor = SYSTEMS[system]["time_to_attractor"]
        transient_timesteps = int(time_to_attractor / dt)
    timesteps = int(duration / dt) + 1
    t = np.linspace(0, time_to_attractor + duration, transient_timesteps + timesteps)
    df = SYSTEMS[system]["df"]
    U = integrate.odeint(df, initial, t, tfirst=True)
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

def plot2d(U, t, color=None):
    if color is None:
        color = np.random.rand(U.shape[1], 3)
    for i, c in enumerate(color):
        plt.plot(t, U[:, i], c=c)
    plt.show()
    return color
