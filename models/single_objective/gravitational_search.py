import numpy as np
import math
from collections import namedtuple
from models.swarm_algorithm import SwarmAlgorithm


GravitationalSearchParams = namedtuple(
    'GravitationalSearchParams',
    ['G0', 'alpha']
)


class GravitationalSearch(SwarmAlgorithm):
    '''
    Gravitational Search algorithm.

    Parameters
    ----------
    D : int
        Search space dimension.
    N : int
        Population size.
    fit_func : callable
        Fitness (objective) function.
    params : GravitationalSearchParams
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
    G0 : float
        The initial value of gravitational constant.
    alpha : float
        Reduction regulation parameter.
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

    def set_params(self, new_params):
        '''
        Initialize the algorithm with a strategy (vector of parameters).

        Parameters
        ----------
        new_params : GravitationalSearchParams

        Returns
        -------
        No value.
        '''
        self.G0 = new_params.G0
        self.alpha = new_params.alpha

    def __get_acceleration(self, M, G, iteration):
        '''
        Computes the acceleration for each object.

        Parameters
        ----------
        M : ndarray
            An array of size N representing object (particles) masses.
        G : float
            Gravitational constant.
        iteration : int
            Current iteration of the optimization process.
        
        Returns
        -------
        ndarray
            An N by D matrix, which represents an array of acceleration vectors
            for each object.
        '''
        final_per = 2 # Drawn from the original paper implementation.
        kbest = final_per + (1 - iteration / self.max_iter) * (100 - final_per)
        kbest = math.trunc(self.N * kbest / 100)
        M_sorted_i = np.argsort(-M)
        E = np.zeros((self.N, self.D))
        
        for i in range(self.N):
            for ii in range(kbest):
                j = M_sorted_i[ii]
                if j != i:
                    R = np.linalg.norm(self.particles[i] - self.particles[j])
                    vec_dist = self.particles[j] - self.particles[i]
                    E[i] += np.random.uniform(size=self.D) * M[j] * vec_dist / (R + 0.001)
        return E * G

    def __move_all(self, a):
        '''
        Updates the positions of all the particles in the swarm in-place.

        Parameters
        ----------
        a : ndarray
            An N by D matrix, which represents an array of acceleration vectors
            for each object.

        Returns
        -------
        No value.
        '''
        self.velocities = np.random.uniform(size=(self.N, self.D)) * self.velocities + a
        self.particles += self.velocities
        self.simplebounds(self.particles)
        for i in range(self.N):
            self.scores[i] = self.fit_func(self.particles[i])
            if self.scores[i] < self.gbest_score:
                self.gbest_score = self.scores[i]
                self.gbest = np.copy(self.particles[i])

    def __mass_calc(self):
        '''
        Calculates object masses based on the fitness-function values.

        Parameters
        ----------
        No parameters.

        Returns
        -------
        ndarray
            An array of size N containing object masses.
        '''
        f_min = np.min(self.scores)
        f_max = np.max(self.scores)

        if f_max == f_min:
            M = np.ones(self.N)
        else:
            M = (self.scores - f_max) / (f_min - f_max)
        return M / np.sum(M)

    def __g_const_calc(self, iteration):
        '''
        Reduces gravitational constant as the iterations' number increases
        (makes the search more accurate).

        Parameters
        ----------
        iteration : int
            Current iteration of the optimization process.
        Returns
        -------
        float
            New value of gravitational constant.
        '''
        # return self.G0 * (1 / (self.eval_num + 1)) ** self.alpha
        return self.G0 * math.exp(-self.alpha * iteration / self.max_iter)

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
            M = self.__mass_calc()
            G = self.__g_const_calc(i + 1)
            a = self.__get_acceleration(M, G, i + 1)
            self.__move_all(a)
            self.eval_num += self.N
            i += 1
        return self.gbest
        