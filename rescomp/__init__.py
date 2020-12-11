__version__ = "0.0.1"
from .ResComp import ResComp
from .DrivenResComp import DrivenResComp
from .chaosode import lorenz, rossler, thomas, random_initial, orbit, plot3d, plot2d, SYSTEMS
from .utils import relerr, accduration, system_fit_error, train_test_orbit, lyapunov
from .complex_networks import barabasi, erdos, random_digraph, watts, geometric
