import numpy as np
from collections import namedtuple
from models.swarm_algorithm import SwarmAlgorithm


PSOParams = namedtuple(
    'PSOParams',
    ['w', 'c1', 'c2']
)


class PSO(SwarmAlgorithm):
    '''
    Particle Swarm Optimization algorithm.

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
    w : float
        Inertial parameter.
    c1 : float
        Cognitive parameter.
    c2 : float
        Social parameter.
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
    velocities : ndarray
        Particles' velocities.
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

        # Compute initial best positions for each particle individually.
        self.pbest = np.copy(self.particles)
        self.pbest_scores = np.copy(self.scores)

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
        self.w = new_params.w
        self.c1 = new_params.c1
        self.c2 = new_params.c2

    def __move_all(self):
        '''
        Updates the positions of all the particles in the swarm in-place.

        Parameters
        ----------
        No parameters.

        Returns
        -------
        No value.
        '''
        self.velocities = self.w * self.velocities
        U1 = np.random.uniform(0, self.c1, (self.N, self.D))
        U2 = np.random.uniform(0, self.c2, (self.N, self.D))
        self.velocities += U1 * (self.pbest - self.particles)
        self.velocities += U2 * (self.gbest - self.particles)
        self.particles += self.velocities
        self.simplebounds(self.particles)

        # Get new objective function values and update best solutions.
        for i in range(self.N):
            self.scores[i] = self.fit_func(self.particles[i])

        better_individual = self.scores < self.pbest_scores
        self.pbest[better_individual] = np.copy(self.particles[better_individual])
        self.pbest_scores[better_individual] = self.scores[better_individual]

        self.update_best()

    # Main loop of the algorithm.
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
        while (i < self.max_iter) and (stag_count < self.stag_iter):
            self.__move_all()
            self.eval_num += self.N
            i += 1
            # Count stagnation iterations (global best doesn't change much).
            if abs(prev_best_score - self.gbest_score) <= self.e:
                stag_count += 1
            elif stag_count > 0:
                stag_count = 0
            prev_best_score = self.gbest_score
        return self.gbest
