import numpy as np
import random
import math


class BatSearch:
    def __init__(self, D, N, fmin, fmax, fit_func, b_range, iter_num, A_0=1, r_0=0.5, sigma=0.01, alfa=0.9, gamma=0.9):
        self.D = D
        self.N = N
        self.fmin = fmin
        self.fmax = fmax
        self.sigma = sigma
        self.alfa = alfa
        self.r_0 = r_0
        self.gamma = gamma
        self.fit_func = fit_func
        self.b_range = b_range
        self.iter_num = iter_num

        # Generating initial population.
        self.bats = np.array([self.__generate_bat() for _ in range(N)])
        self.velocities = np.zeros((N, D))
        self.frequencies = np.zeros(N)
        self.loudness = np.full(N, A_0)
        self.rates = np.full(N, r_0)

        # Calculating scores (fitness) for each agent (nest).
        self.scores = np.array([fit_func(bat) for bat in self.bats])

        # Getting current best.
        self.gbest = np.copy(self.bats[np.ndarray.argmin(self.scores)])
        self.gbest_score = self.fit_func(self.gbest)
    
    # Generates a new bat at random position with each coordinate at range (-b_range; b_range).
    def __generate_bat(self):
        new_bat_coords = np.random.rand(self.D) * 2 * self.b_range - self.b_range
        return new_bat_coords

    def __update_bats_pos(self):
        self.frequencies = self.fmin + (self.fmax - self.fmin) * np.random.randn(self.N)
        self.velocities += np.multiply(self.bats - self.gbest, self.frequencies) 
        self.bats += self.velocities

    def optimize(self):
        t = 0
        gen_count = 1
        while (t < self.iter_num):
            for i in range(self.N):
                self.frequencies[i] = self.fmin + (self.fmax - self.fmin) * np.random.uniform()
                self.velocities[i] += (self.bats[i] - self.gbest) * self.frequencies[i]
                new_solution = self.bats[i] + self.velocities[i]
                self.__simplebounds(new_solution)
                if np.random.rand() > self.rates[i]:
                    new_solution = np.copy(self.gbest)
                    # Perform local search arround the best solution.
                    new_solution += self.sigma * np.random.normal(size=self.D) 
                    new_solution *= np.mean(self.loudness)
                    self.__simplebounds(new_solution)

                new_score = self.fit_func(new_solution)
                if new_score <= self.scores[i] and np.random.rand() < self.loudness[i]:
                    self.bats[i] = np.array(new_solution)
                    self.scores[i] = new_score

                    # Increase rate and reduce loudness.
                    self.loudness[i] *= self.alfa
                    self.rates[i] = self.r_0 * (1 - math.exp(-self.gamma * (gen_count)))
                
                if new_score < self.gbest_score:
                    self.gbest = np.array(new_solution)
                    self.gbest_score = new_score

            t += self.N
            gen_count += 1
        return self.gbest

    # Simple constraint rule for agents' positions.
    def __simplebounds(self, coords):
        for i in range(self.D):
            if coords[i] < -self.b_range:
                coords[i] = -self.b_range
            elif coords[i] > self.b_range:
                coords[i] = self.b_range
        