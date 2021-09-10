#optimizer_controller.py
"""Class to perform hyperparameter optimization on a reservoir computer"""

from ipyparallel import Client
import sherpa
import numpy as np

from .optimizer_systems import get_system, loadprior
from .templates import System
from .optimizer_functions import get_paramlist, create_orbit, vpt, test_all

class ResCompOptimizer:
    """
    A class for easily performing hyperparameter optimization on a ResComp object.
    
    Attributes:
        system (rescomp.optimize.System): the system to get data from
        prediction_type (str): 'continue' or 'random'; the type of predictions to optimize the reservoir computer for
        parallel (bool): whether using parallelization
        results_directory (str): directory to save intermediate optimization results in
        prior (list of parameter dictionaries): sets of hyperparameters known to be good, used in certain types 
        opt_parameters (list of sherpa.Parameter): sherpa parameters to optimize over
        opt_param_names (list of str): names of parameters being optimized over
        study (sherpa.Study): the internal sherpa study object used
        
        res_params (dictionary): any additional parameters to pass to the reservoir computer
        
        If using parallelization:
            dview (ipp.dview): view into ipyparallel nodes
            node_count (int): number of nodes in use
    """
    def __init__(self, system, map_initial, prediction_type, method, res_ode=None,
                add_params=None, rm_params=None, results_directory="", data_directory="",
                parallel=False, parallel_profile=None, **res_params):
        """
        Arguments:
            system (string or rescomp.optimizer.System): the system to use. If not a rescomp.optimizer.System object, will attempt to load one using rescomp.optimizer.get_system(system)
            map_initial (string): initial condition mapping for reservoir computer to use
            prediction_type (string): 'random' or 'continue'; prediction type to use while optimizing.
            method (string): training method; 'standard' or 'augmented'
            
        Optional arguments:
            res_ode (dict->callable): dictionary containing functions 'res_ode' and 'trained_res_ode' to replace the reservoir computer's usual internal ODE
            add_params: list of sherpa.Parameter objects to include in optimization
            rm_params (list of str): names of optimization parameters to remove
            
            results_directory (str): pathname of where to store optimization results. Default will store in current directory.
            data_directory (str): pathname to load additional priors from
            
            parallel (bool): whether to use parallelization. Default false
            parallel_profile (str or None): when using parallelization, the ipyparallel profile to connect to.
            
            All other keyword arguments are passed to the reservoir computers created.
        
        When specifying an alternate reservoir ODE, the dictionary should map 'res_ode' and 'trained_res_ode' to functions that have the same signatures as the corresponding function in ResComp or DrivenResComp.            
        """
        if not isinstance(system, System):
            self.system = get_system(system)
        else:
            self.system = system
        self.prediction_type = prediction_type
        
        self.parallel = parallel
        self.results_directory = results_directory
        
        self.opt_parameters, self.opt_param_names = get_paramlist(self.system, method,
                    add=add_params, remove=rm_params)
        self.prior = loadprior(self.system.name, self.opt_parameters, data_directory)
        self.study = None
        
        self.res_params = {**res_params, 'res_ode':res_ode, 'map_initial':map_initial}
        
        if parallel:
            self._initialize_parallelization(parallel_profile)
            
    def run_optimization(self, opt_ntrials, vpt_reps, algorithm='gpyopt', max_stderr=None, sherpa_dashboard=False, raise_err=False):
        """Runs the optimization process.
        
        Arguments:
            opt_ntrials (int): number of hyperparameter configurations to attempt
            vpt_reps (int): number of times to try each parameter set
            algorithm (str or sherpa.Algorithm): the algorithm to use for optimizing. Must be either a sherpa.algorithms.Algorithm object or one of 'grid_search', 'random_search', 'gpyopt', 'successive_halving', 'local_search', or 'population'. Note that local_search will ignore opt_ntrials, and grid_search will only repeat approximately opt_ntrials times, due to their implementations.
            max_stderr (float or None): if not None, ensures that the standard error for the mean is at most this value for each hyperparameter configuration.
            sherpa_dashboard (bool): whether to use the sherpa dashboard. Default false.
            raise_err (bool): whether errors occuring during VPT calculations should be raised; otherwise are suppressed and a VPT of -1 is reported"""
        self._initialize_sherpa(opt_ntrials, algorithm, sherpa_dashboard)
        for trial in self.study:
            try:
                exp_vpt, stderr = self.run_single_vpt_test(vpt_reps, trial.parameters, raise_err=raise_err)
            except Exception as e:
                #print relevant information for debugging
                print("Trial parameters at error:", trial.parameters)
                print("Other parameters:", self.res_params)
                raise e
            self.study.add_observation(trial=trial,
                              objective=exp_vpt,
                              context={'vpt_stderr':stderr})
            self.study.finalize(trial)
            self.study.save(self.results_directory)
    
    def run_tests(self, test_ntrials, lyap_reps=20, parameters=None):
        """
        Runs tests using the given parameters.
        If not passed, uses the optimized hyperparameters.
        
        Tests the reservoir computer for the following quantities:
            -continue prediction vpt
            -random prediction vpt
            -Lyapunov exponent of reservoir's recreation of the system
            For systems governed by a differential equation:
            -derivative fit of continued prediction
            -derivative fit of random prediction
        All of these results are returned in a dictionary mapping the names to the attributes.
        
        Arguments:
            test_ntrials (int): number of times to test the parameter set
            lyap_reps (int): number of repetitions to use when approximating the Lyapunov exponent
            parameters (dict): the hyperparameter set to use. If not specified, uses the best result from optimizing.
        """
        if parameters is None:
            parameters = self.get_best_result()
            
        if self.parallel:
            results, _ = self._run_n_times_parallel(test_ntrials, test_all,
                        self.system, lyap_reps=lyap_reps, **self.res_params, **parameters)
            #Collapse into single list of outputs
            results = [item for sublist in results for item in sublist]
        else:
            results = self._run_n_times(test_ntrials, test_all,
                        self.system, lyap_reps=lyap_reps, **self.res_params, **parameters)
        
        #Collect results into a dictionary
        results_dict = {name:[] for name in {'continue','random','lyapunov','cont_deriv_fit','rand_deriv_fit'}}
        for item in results:
            cont_vpt, rand_vpt, lyap, cont_df, rand_df = item
            results_dict["continue"].append(cont_vpt)
            results_dict["random"].append(rand_vpt)
            results_dict["lyapunov"].append(lyap)
            if self.system.is_diffeq:
                results_dict["cont_deriv_fit"].append(cont_df)
                results_dict["rand_deriv_fit"].append(rand_df)
        return results_dict
    
    def generate_orbits(self, n_orbits, parameters=None, return_rescomp=False):
        """
        Trains a reservoir computer and has it predict, using the given hyperparameters
        If parameters are not specified, uses the optimized hyperparameters.
        
        Arguments:
            n_orbits (int): number of predicitons to make using the parameter set
            parameters (dict): the hyperparameter set to use. If not specified, uses the best result from optimizing.
        
        Returns:
            a list of orbit data, where each entry is a tuple consisting of (tr, Utr, ts, Uts, pre).
        """
        if parameters is None:
            parameters = self.get_best_result()
            
        if self.parallel:
            results, _ = self._run_n_times_parallel(n_orbits, create_orbit,
                        self.system, self.prediction_type, **self.res_params, **parameters, return_rescomp=return_rescomp)
            #Collapse into single list of outputs
            results = [item for sublist in results for item in sublist]
        else:
            results = self._run_n_times(n_orbits, create_orbit,
                        self.system, self.prediction_type, **self.res_params, **parameters, return_rescomp=return_rescomp)
        
        return results
    
    def run_single_vpt_test(self, vpt_reps, trial_params, max_stderr=None, raise_err=False):
        """Returns the mean and standard error of valid prediction time (VPT) resulting from the current and specified parameters."""
        
        if max_stderr is not None:
            if max_stderr <= 0:
                raise ValueError("maximum standard error must be positive!")
        
        total_count = 0
        vpts = []
        try:
            while True:
                if self.parallel:
                    new_vpts, ct = self._run_n_times_parallel(vpt_reps, vpt,
                                self.system, self.prediction_type, **self.res_params, **trial_params)
                else:
                    new_vpts = self._run_n_times(vpt_reps, vpt,
                                self.system, self.prediction_type, **self.res_params, **trial_params)
                    ct = vpt_reps
                
                total_count += ct
                vpts += new_vpts
                if max_stderr is None:
                    break
                else:
                    stderr = np.std(vpts)/np.sqrt(total_count)
                    if stderr < max_stderr:
                        break
        except Exception as e:
            if raise_err:
                raise
            else:
                print(f"{type(e).__name__}: {str(e)}")
                print(f"Trial parameters: {trial_params}:")
                print("Returning VPT as -1")
                return -1, 0
        
        return np.mean(vpts), np.std(vpts)/np.sqrt(total_count)
        
    def get_best_result(self):
        """Returns the best parameter set found in the previous optimization attempt."""
        if self.study is None:
            raise RuntimeError("must run optimization before getting results!")
        result = self.study.get_best_result()
        #Clean non-parameters from the dictionary
        return {key:result[key] for key in result.keys() if key in self.opt_param_names}
    
    #### Internal functions ####
        
    def _run_n_times(self, n, func, *args, **kwargs):
        """
        Calls func(*args, **kwargs) n times, and returns the result as a list.
        """
        return [func(*args, **kwargs) for _ in range(n)]
        
    def _run_n_times_parallel(self, n, func, *args, **kwargs):
        """
        Calls func(*args, **kwargs) (at least) n times total between the ipyparallel nodes, and returns the result as a list, as well as the total number of calls to the function.
        The function is called the same number of times on each node, and guaranteed to be called at least a total number of n times, although generally will be slightly more.
        """
        run_ct = int(np.ceil(n / self.node_count))
        result = self.dview.apply(lambda k, *args, **kwargs: [func(*args, **kwargs) for _ in range(k)],
                        run_ct, *args, **kwargs)
        return result, run_ct * self.node_count
        
    def _initialize_parallelization(self, parallel_profile):
        """Helper function to set up parallelization.
        Connects to the ipyparallel client and initializes an engine on each node."""
        if not self.parallel:
            #this shouldn't happen really ever
            raise RuntimeError("parallelization cannot be set up if not enabled")
        client = Client(profile=parallel_profile)
        self.dview = client[:]
        self.dview.use_dill()
        self.dview.block = True #possibly can remove, but would require modifying other parts
        self.node_count = len(client.ids)
        print(f"Using multithreading; running on {self.node_count} engines.")
    
    def _initialize_sherpa(self, opt_ntrials, algorithm, sherpa_dashboard=False):
        """Initializes the sherpa study used internally"""
        #Initialize the algorithm if needed
        if isinstance(algorithm, str):
            algorithm = algorithm.lower()
        if isinstance(algorithm, sherpa.algorithms.Algorithm):
            algorithm_obj = algorithm
        elif algorithm == 'grid_search':
            gridpoints = int(np.round(opt_ntrials**(1/len(self.opt_parameters))))
            algorithm_obj = sherpa.algorithms.GridSearch(num_grid_points = gridpoints)
        elif algorithm == 'random_search':
            algorithm_obj = sherpa.algorithms.RandomSearch(max_num_trials=opt_ntrials)
        elif algorithm == 'gpyopt':
            algorithm_obj = sherpa.algorithms.GPyOpt(max_num_trials=opt_ntrials, initial_data_points=self.prior)
        elif algorithm == 'successive_halving':
            algorithm_obj = sherpa.algorithms.SuccessiveHalving(max_finished_configs=opt_ntrials)
        elif algorithm == 'local_search':
            if len(self.prior)==0:
                raise ValueError("LocalSearch algorithm requires a non-empty prior")
            algorithm_obj = sherpa.algorithms.LocalSearch(seed_configuration=self.prior[0])
        elif algorithm == 'population':
            population_size = 20
            num_generations = int(np.ceil((opt_ntrials - population_size)/(population_size / 5)))
            if num_generations < 2:
                num_generations = 2
            parameter_range = {var.name:var.range for var in self.opt_parameters if isinstance(var, sherpa.Continuous)}
            algorithm_obj = sherpa.algorithms.PopulationBasedTraining(num_generations=num_generations,
                        population_size=population_size,parameter_range=parameter_range)
        else:
            raise ValueError(f"algorithm parameter must be a sherpa.algorithms.Algorithm object or one of 'grid_search', 'random_search', 'gpyopt', 'successive_halving', 'local_search', or 'population', not {algorithm}.")
        
        #Initialize the study
        self.study = sherpa.Study(parameters=self.opt_parameters,
                         algorithm=algorithm_obj,
                         disable_dashboard=(not sherpa_dashboard),
                         lower_is_better=False)
                         
