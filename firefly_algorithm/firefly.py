import numpy as np
import random as rand


# Class, which represents a single firefly in a swarm.
class Firefly:
    # D - dimension of a search space;
    # p_range - range of coordinates' values (from -range to range);
    # fit_func - object function.
    def __init__(self, D, p_range, fit_func):
        self.D = D
        self.p_range = p_range
        self.coordinates = np.random.rand(D) * 2 * p_range - p_range
        self.score = fit_func(self.coordinates)

    # Gets attractiveness for another firefly.
    def __beta(self, firefly, beta_zero, gamma):
        # Euclidian distance between two fireflies.
        distance = np.linalg.norm(self.coordinates - firefly.coordinates)
        # We can rewrite the attractiveness function as follows:
        #   beta_zero * np.exp(-gamma * distance ** 2)
        # But to improve the perfomance, we define it as:
        return beta_zero / (1 + gamma * distance ** 2)

    # Move a firefly towards another one.
    def move_towards_firefly(self, firefly, beta_zero, alfa, gamma, lambd, gbest_pos, fit_func):
        # Get the attractiveness of the firefly we want to fly to.
        current_beta = self.__beta(firefly, beta_zero, gamma)

        # Get vector difference of fireflies' coordinates.
        diff = self.coordinates - firefly.coordinates

        # Get vector of random values in range (-1; 1).
        urand_n = np.multiply(
            np.power(np.full(self.D, -1), np.random.randint(2, size = self.D)), 
            np.random.rand(self.D)
        )

        # Update coordinates according to the flight formula.
        self.coordinates = firefly.coordinates + current_beta * diff + alfa * urand_n

        # Modifications:
        # Get vector of random values in range (-1; 1).
        urand_n = np.multiply(
            np.power(np.full(self.D, -1), np.random.randint(2, size = self.D)), 
            np.random.rand(self.D)
        )

        # Add an extra term to the flight formula, which uses global best solution
        # to improve the efficiency.
        self.coordinates += lambd * np.multiply(urand_n, self.coordinates - gbest_pos)

        # Calculate the fit score (brightness) of the firefly based on it's new position.
        self.score = fit_func(self.coordinates)
