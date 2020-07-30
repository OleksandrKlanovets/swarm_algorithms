import numpy as np
import random
import math
from collections import namedtuple
from models.swarm_algorithm import SwarmAlgorithm


BatSearchParams = namedtuple(
    'BatSearchParams',
    ['fmin', 'fmax', 'sigma', 'A_0', 'r_0', 'alpha', 'gamma']
)


class BatSearch(SwarmAlgorithm):
    '''
    Bat Search algorithm.

    Parameters
    ----------
    D : int
        Search space dimension.
    N : int
        Population size.
    fit_func : callable
        Fitness (objective) function.
    params : BatSearchParams
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
    fmin : float
        Minimal frequency.
    fmax : float
        Maximum frequency.
    sigma : float
        Step.
    A_0 : float
        Initial loudness.
    r_0 : float
        Initial rate of pulse emission.
    alpha : float
        A coefficient used to decrease loudness.
    gamma : float
        A coefficient used to decrease the rate of pulse emission.
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
    eval_num : int
        The number of fitness function evaluations.
    '''
    def __init__(self, D, N, fit_func, params, bounds, seed=None, max_iter=100,
                 stag_iter=100, e=0.00001):
        super().__init__(D, N, fit_func, params, bounds, seed, max_iter,
                         stag_iter, e)

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
        SwarmAlgorithm.set_population(self, new_population)

        self.velocities = np.zeros((self.N, self.D))
        self.frequencies = np.zeros(self.N)
        self.loudness = np.full(self.N, self.A_0)
        self.rates = np.full(self.N, self.r_0)

    def set_params(self, new_params):
        '''
        Initialize the algorithm with a strategy (vector of parameters).

        Parameters
        ----------
        new_params : BatSearchParams

        Returns
        -------
        No value.
        '''
        self.fmin = new_params.fmin
        self.fmax = new_params.fmax
        self.sigma = new_params.sigma
        self.A_0 = new_params.A_0
        self.r_0 = new_params.r_0
        self.alpha = new_params.alpha
        self.gamma = new_params.gamma

    def __move_all(self):
        '''
        Updates the positions of all the bats in the swarm.

        Parameters
        ----------
        No parameters.

        Returns
        -------
        No value.
        '''
        freq_range = self.fmax - self.fmin
        rand_coef = np.random.uniform(low=0, high=1, size=(self.N, 1))
        self.frequencies = self.fmin + freq_range * rand_coef
        self.velocities += (self.particles - self.gbest) * self.frequencies
        new_solutions = self.particles + self.velocities
        self.simplebounds(new_solutions)
        return new_solutions

    def __local_search(self, new_solutions):
        '''
        Performs local search around the global best particle.

        Parameters
        ----------
        new_solutions : ndarray
            An array of D-dimensional vectors, which represent new solutions.

        Returns
        -------
        No value.
        '''
        apply_indices = np.random.rand(self.N) > self.rates
        apply_size = np.sum(apply_indices) 

        rand_coefs = np.random.normal(size=(apply_size, self.D))
        new_solutions[apply_indices] = np.copy(self.gbest)
        new_solutions[apply_indices] += self.sigma * rand_coefs
        new_solutions[apply_indices] *= np.mean(self.loudness)
        self.simplebounds(new_solutions[apply_indices])

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
        stag_count = 0
        prev_best_score = self.gbest_score

        # MAIN LOOP
        while (i < self.max_iter) and (stag_count < self.stag_iter):
            new_solutions = self.__move_all()
            self.__local_search(new_solutions)
            new_scores = np.array([self.fit_func(s) for s in new_solutions])
            scores_condition = new_scores < self.scores
            loudness_condition = np.random.rand(self.N) < self.loudness
            apply_indices = np.logical_and(scores_condition, loudness_condition)
            self.particles[apply_indices] = new_solutions[apply_indices]
            self.scores[apply_indices] = new_scores[apply_indices]

            # Increase rate and reduce loudness.
            self.loudness[apply_indices] *= self.alpha
            self.rates[apply_indices] = self.r_0 * (1 - math.exp(-self.gamma * (i + 1)))
                
            self.update_best()
            self.eval_num += self.N
            i += 1

            # Count stagnation iterations (global best doesn't change much).
            if abs(prev_best_score - self.gbest_score) <= self.e:
                stag_count += 1
            elif stag_count > 0:
                stag_count = 0
            prev_best_score = self.gbest_score
        return self.gbest
