import numpy as np
import math


class CuckooSearch:
    # D - dimension of the search space;
    # N - number of particles to generate;
    # xi - fraction of nests to abandon;
    def __init__(self, D, N, xi, fit_func, c_range, iter_num, beta=3/2):
        self.D = D
        self.N = N
        self.xi = xi
        self.beta = beta
        self.fit_func = fit_func
        self.c_range = c_range
        self.iter_num = iter_num
        
        # Generating initial population.
        self.nests = np.array([self.__generate_nest() for _ in range(N)])

        # Calculating scores (fitness) for each agent (nest).
        self.scores = float('inf') * np.ones(self.N)

        # Getting current best.
        self.gbest = self.__get_best_nest(self.nests)
        self.gbest_score = self.fit_func(self.gbest)

    # Performs main optimization process.
    def optimize(self):
        t = 0
        while t < self.iter_num:
            new_nests = self.__get_new_solutions()
            self.__get_best_nest(new_nests)

            new_nests = self.__empty_nests()
            current_best = self.__get_best_nest(new_nests)
            current_best_score = self.fit_func(current_best)

            if current_best_score < self.gbest_score:
                self.gbest = np.copy(current_best)
                self.gbest_score = current_best_score

            t += 2 * self.N
        
        return self.gbest

    # Generates new nest at random position with each coordinate at range (-c_range; c_range).
    def __generate_nest(self):
        new_nest_coords = np.random.rand(self.D) * self.c_range
        new_nest_coords *= np.power(np.full(self.D, -1), np.random.randint(2, size = self.D))
        return new_nest_coords

    # Evaluates new solutions and gets current best.
    def __get_best_nest(self, new_nests):
        # Evaluating new solutions found by cuckoo and replace old ones if necessary.
        for i in range(self.N):
            new_score = self.fit_func(new_nests[i])
            if new_score < self.scores[i]:
                self.scores[i] = new_score
                self.nests[i] = new_nests[i]

        # Return nest with minimal score.
        return np.copy(self.nests[np.argmin(self.scores)])

    # Gets new solutions (cuckoos) by performing Levy flights.
    def __get_new_solutions(self):
        sigma = math.gamma(1 + self.beta) * math.sin(math.pi * self.beta / 2)
        sigma /= (math.gamma((1 + self.beta) / 2) * self.beta * 2 ** ((self.beta - 1) / 2))
        sigma **= (1 / self.beta)

        new_solutions = np.copy(self.nests)

        for i in range(self.N):
            u = np.random.randn(self.D) * sigma
            v = np.random.randn(self.D)
            step = np.divide(u, np.power(np.abs(v), 1 / self.beta))

            stepsize = 0.01 * np.multiply(step, new_solutions[i] - self.gbest)

            new_solutions[i] += np.random.randn(self.D) * stepsize
        
        return new_solutions

    # Replace a fraction of nests by generating new solutions.
    def __empty_nests(self):
        # Decision vector (is discovered).
        decisions = np.random.choice(2, self.N, p=[self.xi, 1 - self.xi])

        stepsize = self.nests[np.random.permutation(self.N)]
        stepsize -= self.nests[np.random.permutation(self.N)]
        stepsize *= np.random.rand()

        new_nests = self.nests + np.multiply(stepsize, decisions[:, np.newaxis])

        for i in range(self.N):
            self.__simplebounds(new_nests[i])
        
        return new_nests

    # Simple constraint rule for agents' positions.
    def __simplebounds(self, coords):
        for i in range(self.D):
            if coords[i] < -self.c_range:
                coords[i] = -self.c_range
            elif coords[i] > self.c_range:
                coords[i] = self.c_range
