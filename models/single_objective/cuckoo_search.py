import numpy as np
import math
from collections import namedtuple
from models.swarm_algorithm import SwarmAlgorithm


CuckooSearchParams = namedtuple(
    'CuckooSearchParams',
    ['xi', 'alpha']
)


class CuckooSearch(SwarmAlgorithm):
    '''
    Cuckoo Search algorithm.

    Parameters
    ----------
    D : int
        Search space dimension.
    N : int
        Population size.
    fit_func : callable
        Fitness (objective) function.
    params : CuckooSearchParams
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
        by equal or more than a given tolerance e. If the number is exceeded, 
        the optimization process stagnation occurs and the algorithm stops.
    e : float, optional, default=1e-5
        Tolerance.

    Attributes
    ----------
    xi : float
        Fraction of nests to abandon.
    alpha : float
        Step size for Levy flights.
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

        # Levy flight's coefficient calculation.
        self.beta = 1.5
        self.sigma = math.gamma(1 + self.beta) 
        self.sigma *= math.sin(math.pi * self.beta / 2)
        # math.gamma((1 + self.beta) / 2) * self.beta * 2 ** ((self.beta - 1) / 2)
        denominator = math.gamma((1 + self.beta) / 2)
        denominator *= self.beta * 2 ** ((self.beta - 1) / 2)
        self.sigma /= denominator
        self.sigma **= (1 / self.beta)

    def set_params(self, new_params):
        '''
        Initialize the algorithm with a strategy (vector of parameters).

        Parameters
        ----------
        new_params : CuckooSearchParams

        Returns
        -------
        No value.
        
        '''
        self.xi = new_params.xi
        self.alpha = new_params.alpha

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
        # Initialize stagnating iterations counter.
        stag_count = 0
        prev_best_score = self.gbest_score

        # MAIN LOOP
        while i < self.max_iter and stag_count < self.stag_iter:
            new_nests = self.__get_new_solutions()
            self.__eval_solutions(new_nests)
            new_nests = self.__empty_nests()
            self.__eval_solutions(new_nests)
            self.update_best()
            self.eval_num += 2 * self.N
            i += 1

            # Count stagnation iterations (global best doesn't change much).
            if abs(prev_best_score - self.gbest_score) <= self.e:
                stag_count += 1
            elif stag_count > 0:
                stag_count = 0
            prev_best_score = self.gbest_score
        return self.gbest

    def __eval_solutions(self, new_nests):
        '''
        Computes the value of the fitness-function for the new nests and
        updates the swarm scores if necessary.

        Parameters
        ----------
        new_nests : ndarray
            An array of D-dimensional vectors, which represent nests'
            positions.

        Returns
        -------
        No value.
        '''
        new_scores = np.zeros(self.N)
        for i in range(self.N):
            new_scores[i] = self.fit_func(new_nests[i])
        upd_indexes = new_scores < self.scores
        self.scores[upd_indexes] = new_scores[upd_indexes]
        self.particles[upd_indexes] = new_nests[upd_indexes]

    def __get_new_solutions(self):
        '''
        Gets new solutions by performing Levy flights.

        Parameters
        ----------
        No parameters.

        Returns
        -------
        ndarray
            An array of D-dimensional vectors, which represent new nests'
            positions.
        '''
        new_solutions = np.copy(self.particles)
        u = np.random.normal(size=(self.N, self.D)) * self.sigma
        v = np.random.normal(size=(self.N, self.D))
        step = u / np.abs(v) ** (1 / self.beta)
        stepsize = self.alpha * step * (new_solutions - self.gbest)
        new_solutions += np.random.normal(size=(self.N, self.D)) * stepsize
        self.simplebounds(new_solutions)
        return new_solutions

    def __empty_nests(self):
        '''
        Replaces a fraction of nests (determined by xi) by generating 
        new solutions.

        Parameters
        ----------
        No parameters.

        Returns
        -------
        ndarray
            An array of D-dimensional vectors, which represent new nests'
            positions.
        '''
        # Decision vector (is discovered).
        decisions = np.random.uniform(size=self.N) > self.xi

        stepsize = self.particles[np.random.permutation(self.N)]
        stepsize -= self.particles[np.random.permutation(self.N)]
        stepsize *= np.random.uniform(0, 1, self.D)

        new_nests = self.particles + stepsize * decisions[:, np.newaxis]
        self.simplebounds(new_nests)
        return new_nests
