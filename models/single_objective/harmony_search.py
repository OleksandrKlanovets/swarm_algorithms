import numpy as np
from collections import namedtuple
from models.swarm_algorithm import SwarmAlgorithm


HarmonySearchParams = namedtuple(
    'HarmonySearchParams',
    ['r_accept', 'r_pa', 'b_range']
)


class HarmonySearch(SwarmAlgorithm):
    '''
    Harmony Search algorithm.

    Parameters
    ----------
    D : int
        Search space dimension.
    N : int
        Population size.
    fit_func : callable
        Fitness (objective) function.
    params : PSOParams
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
    r_accept : float
        Memory accepting rate.
    r_pa : float
        Pitch adjusting rate.
    b_range : float
        Pitch bandwidth.
    particles : ndarray
        An N by D array representing the swarm of N particles.
    scores : ndarray
        An array of size N representing the value of the fitness function
        for each particle.
    gbest : ndarray
        A D-dimensional vector representing the position of the current 
        global best particle.
    gbest_score : float
        The value of the fitness function for the current global best particle.
    pbest : ndarray
        An N by D array representing the best positions found by each particle
        individually.
    pbest_scores : ndarray
        Fitness function values for positions in pbest attribute.
    eval_num : int
        The number of fitness function evaluations.
    '''
    def __init__(self, D, N, fit_func, params, bounds, seed=None, max_iter=1000,
                 stag_iter=1000, e=0.00001):
        super().__init__(D, N, fit_func, params, bounds, seed, max_iter,
                         stag_iter, e)

    def set_params(self, new_params):
        '''
        Initialize the algorithm with a strategy (vector of parameters).

        Parameters
        ----------
        new_params : PSOParams

        Returns
        -------
        No value.
        '''
        self.r_accept = new_params.r_accept
        self.r_pa = new_params.r_pa
        self.b_range = new_params.b_range

    def __form_new_harmony(self):
        '''
        Generates new solutions by adjusting frequencies, randomizing, etc.

        Parameters
        ----------
        No parameters.

        Returns
        -------
        No value.
        '''
        new_harmonies = np.zeros((self.N, self.D))
        decisions = np.random.uniform(size=(self.N, self.D)) > self.r_accept
        inv_decisions = ~decisions

        coords_range = self.u_bounds - self.l_bounds
        new_coordinates = self.l_bounds + np.random.rand(self.N, self.D) * coords_range

        new_harmonies[decisions] = new_coordinates[decisions]
        random_harmonies = self.particles[np.random.choice(self.N, self.N)]
        new_harmonies[inv_decisions] = random_harmonies[inv_decisions]
        
        adjust_probs = np.random.uniform(size=(self.N, self.D)) > self.r_pa
        adjust_indices = np.logical_and(decisions, adjust_probs)
        adjust_shape = new_harmonies[adjust_indices].shape
        adjust_step = self.b_range * np.random.normal(size=adjust_shape)
        adjust_step *= np.random.choice([1, -1])
        new_harmonies[adjust_indices] += adjust_step

        for i in range(self.N):
            new_score = self.fit_func(new_harmonies[i])
            worst_index = np.argmax(self.scores)
            if new_score < self.scores[worst_index]:
                self.scores[worst_index] = new_score
                self.particles[worst_index] = new_harmonies[i]

            if new_score < self.gbest_score:
                self.gbest = np.copy(new_harmonies[i])
                self.gbest_score = new_score

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
        i = 0
        while (i < self.max_iter):
            self.__form_new_harmony()
            self.eval_num += self.N
            i += 1
        return self.gbest
        