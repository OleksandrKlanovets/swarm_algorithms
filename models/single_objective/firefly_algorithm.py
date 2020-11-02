import numpy as np
from collections import namedtuple
from models.swarm_algorithm import SwarmAlgorithm


FireflyAlgorithmParams = namedtuple(
    'FireflyAlgorithmParams',
    ['beta_0', 'alpha_0', 'alpha_inf', 'gamma', 'lambd']
)


class FireflyAlgorithm(SwarmAlgorithm):
    '''
    Firefly algorithm.

    Parameters
    ----------
    D : int
        Search space dimension.
    N : int
        Population size.
    fit_func : callable
        Fitness (objective) function.
    params : FireflyAlgorithmParams
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
    beta_0 : float
        Zero distance attractiveness.
    alpha_0 : float
        Initial value of randomization coefficient.
    alpha_inf : float
        Final value of randomization coefficient.
    alpha : float
        Randomization coefficient.
    gamma : float
        Light absorption coefficient.
    lambd : float
        Randomization coefficient determining the weight of the third component
        of the update rule, which coordinates the movement towards
        the global best solution.
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
                 stag_iter=100, e=0.0001):
        super().__init__(D, N, fit_func, params, bounds, seed, max_iter,
                         stag_iter, e)

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
        SwarmAlgorithm.reset(self)
        self.alpha = self.alpha_0

    def set_params(self, new_params):
        '''
        Initialize the algorithm with a strategy (vector of parameters).

        Parameters
        ----------
        new_params : FireflyAlgorithmParams

        Returns
        -------
        No value.
        '''
        self.beta_0 = new_params.beta_0
        self.alpha_0 = new_params.alpha_0
        self.alpha_inf = new_params.alpha_inf
        self.gamma = new_params.gamma
        self.lambd = new_params.lambd

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
            self.__reduce_alpha(i)

            # Rank the fireflies.
            sorted_indices = np.argsort(self.scores)
            self.scores = self.scores[sorted_indices]
            self.particles = self.particles[sorted_indices]
            self.gbest = np.copy(self.particles[0])
            self.gbest_score = self.scores[0]
            self.__move_all()

            for j in range(self.N):
                self.scores[j] = self.fit_func(self.particles[j])

            self.eval_num += self.N
            i += 1

            # Count stagnation iterations (global best doesn't change much).
            if abs(prev_best_score - self.gbest_score) <= self.e:
                stag_count += 1
            elif stag_count > 0:
                stag_count = 0

            prev_best_score = self.gbest_score
        return self.gbest

    def __move_first_to_second(self, f1, f2):
        '''
        Moves the first firefly (first argument) towards the second one (second
        argument).

        Parameters
        ----------
        f1 : ndarray
            A vector representing firefly to move.
        f2 : ndarray
            A vector representing firefly to move to.

        Returns
        -------
        No value.
        '''
        # Euclidian distance between two fireflies.
        r = np.linalg.norm(f1 - f2)

        # Get the attractiveness of the firefly we want to fly to.
        # beta = self.beta_0 * math.exp(-self.gamma * r ** 2)
        beta = self.beta_0 / (1 + self.gamma * r ** 2)

        diff = f2 - f1
        urand_1 = np.random.randn(self.D)

        # Update coordinates according to the flight formula.
        f1 += beta * diff + self.alpha * urand_1

        # Modifications:
        # Add an extra term to the flight formula, which uses global best
        # solution to improve the efficiency.
        urand_2 = np.random.randn(self.D)
        f1 += self.lambd * urand_2 * (self.gbest - f1)

    def __move_all(self):
        '''
        Updates the positions of all the fireflies in the swarm.

        Parameters
        ----------
        No parameters.

        Returns
        -------
        No value.
        '''
        for i in range(self.N):
            for j in range(self.N):
                if self.scores[j] < self.scores[i]:
                    # Move less attractive to more attractive.
                    self.__move_first_to_second(
                        self.particles[i],
                        self.particles[j]
                    )
        self.simplebounds(self.particles)

    def __reduce_alpha(self, iteration):
        '''
        Reduces alfa as the number of iterations increases.
        Optional modification.

        Parameters
        ----------
        No parameters.

        Returns
        -------
        No value.
        '''
        alpha_range = self.alpha_0 - self.alpha_inf
        self.alpha = self.alpha_inf + alpha_range * np.exp(-iteration)
