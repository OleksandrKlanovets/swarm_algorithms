from abc import ABC, abstractmethod
import numpy as np


class SwarmAlgorithm(ABC):
    '''
    A base abstract class for different swarm algorithms.

    Parameters
    ----------
    D : int
        Search space dimension.
    N : int
        Population size.
    fit_func : callable
        Fitness (objective) function or a function returning multiple values
        corresponding to different objectives (for multi-objective problems).
    params : array_like
        Model behavioral parameters.
    bounds : ndarray
        A 2 by D matrix containing lower and upper bounds of the search space 
        for each dimension.
    seed : int, optional, default=None
        Random generator seed.
    max_iter : int, optional, default=100
        Maximum number of iterations (generations).
    stag_iter : int, optional, default=100
        Specifies the allowed number of iterations without solution improvement
        by equal or more than a given tolerance. If the number is exceeded, 
        the optimization process stagnations occurs and the algorithm stops.
    e : float, optional, default=1e-5
        Tolerance.

    Attributes
    ----------
    particles : ndarray
        An N by D array representing the swarm of N particles.
    scores : ndarray
        An array of size N representing the value of the fitness function
        for each particle.
    gbest : ndarray
        The D-dimensional vector representing the position of the current 
        global best particle.
    gbest_score : float
        The value of the fitness function for the current global best particle.
    eval_num : int
        The number of fitness function evaluations.
    '''
    def __init__(self, D, N, fit_func, params, bounds, seed=None, max_iter=100,
                 stag_iter=100, e=1e-5):
        self.D = D
        self.N = N

        # Initialize problem parameters.
        self.fit_func = fit_func
        self.l_bounds = bounds[0]
        self.u_bounds = bounds[1]

        # Behavioural parameters' initialization.
        self.set_params(params)

        # Initializing the Numpy random numbers generator to reproduce results
        # of the optimization processes.
        self.seed = seed

        # Stopping criteria.
        self.max_iter = max_iter
        self.stag_iter = stag_iter
        self.e = e

        self.reset()

    @abstractmethod
    def set_params(self, new_params):
        '''
        Initialize the algorithm with a strategy (vector of parameters).

        Parameters
        ----------
        new_params : array_like

        Returns
        -------
        No value.
        
        '''
        pass

    def reset(self):
        '''
        Resets the algorithm state.

        Parameters
        ----------
        No parameters.

        Returns
        -------
        No value.
        '''
        if self.seed is not None:
            np.random.seed(self.seed)

        # Generate initial population and particles' velocities.
        self.set_population([self.generate_particle()
                             for _ in range(self.N)])

    def generate_particle(self):
        '''
        Generates a swarm particle within bounds.

        Parameters
        ----------
        No parameters.

        Returns
        -------
        ndarray
            A vector of size D representing particle's coordinates.
        '''
        coords_range = self.u_bounds - self.l_bounds
        return self.l_bounds + np.random.uniform(size=self.D) * coords_range

    def set_population(self, new_population):
        '''
        Sets a population with a pre-generated one.

        Parameters
        ----------
        new_population: array_like
            A matrix with dimensions N by D, which represents the coordinates
            of each particle.

        Returns
        -------
        No value.
        '''
        self.eval_num = self.N

        self.N = len(new_population)
        self.particles = np.copy(new_population)
        self.scores = np.array([self.fit_func(p) for p in self.particles])

        # Initializing current best.
        gbest_index = np.ndarray.argmin(self.scores)
        self.gbest = np.copy(self.particles[gbest_index])
        self.gbest_score = self.scores[gbest_index]

    @abstractmethod
    def optimize(self):
        '''
        Main loop of the algorithm.

        Parameters
        ----------
        No parameters.

        Returns
        -------
        ndarray
            The coordinates of the global best particle at the end of
            the optimization process. 
        '''
        pass

    def update_best(self):
        '''
        Updates global best particle if needed.

        Parameters
        ----------
        No parameters.

        Returns
        -------
        No value.
        '''
        current_best_index = np.argmin(self.scores)
        current_best = self.particles[current_best_index]
        current_best_score = self.scores[current_best_index]

        if current_best_score < self.gbest_score:
            self.gbest = np.copy(current_best)
            self.gbest_score = current_best_score

    def simplebounds(self, coords):
        '''
        Simple constraint rule for particles' positions 
        (in-place coordinate modification).

        Parameters
        ----------
        coords: ndarray
            An array of particles to apply the rule to.

        Returns
        -------
        No value.
        '''
        l_bounds_tiled = np.tile(self.l_bounds, [coords.shape[0], 1])
        u_bounds_tiled = np.tile(self.u_bounds, [coords.shape[0], 1])
        lower_bound_indexes = coords < self.l_bounds
        upper_bound_indexes = coords > self.u_bounds
        coords[lower_bound_indexes] = l_bounds_tiled[lower_bound_indexes]
        coords[upper_bound_indexes] = u_bounds_tiled[upper_bound_indexes]

    def info(self):
        '''
        Returns basic information about the algorithm state in a
        human-readable representation.

        Parameters
        ----------
        No parameters.

        Returns
        -------
        str
            Information about current best position, score and
            current number of fitness-function evaluations.
        '''
        info = f'Algorithm: {type(self).__name__}\n'
        info += f'Best position: {self.gbest}\n'
        info += f'Best score: {self.gbest_score}\n'
        info += f'Fitness function evaluatiions number: {self.eval_num}'

        return info
