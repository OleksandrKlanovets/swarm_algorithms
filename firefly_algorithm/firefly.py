import numpy as np
import random as rand


class Firefly:
    def __init__(self, D, p_range=50):
        self.D = D
        self.p_range = p_range
        self.coordinates = np.random.rand(D) * p_range
        self.coordinates *= np.power(np.full(D, -1), np.random.randint(2, size = D))

    # Gets attractiveness for another firefly.
    def __beta(self, firefly, beta_zero, gamma):
        distance = np.linalg.norm(self.coordinates - firefly.coordinates)
        # return beta_zero / (1 + gamma * distance * distance)
        return beta_zero * np.exp(-gamma * distance * distance)

    def move_towards_firefly(self, firefly, beta_zero, alfa, gamma, lambd, gbest_pos):
        current_beta = self.__beta(firefly, beta_zero, gamma)
        diff = self.coordinates - firefly.coordinates
        urand_n = np.multiply(
            np.power(np.full(self.D, -1), np.random.randint(2, size = self.D)), 
            np.random.rand(self.D)
        )
        self.coordinates = firefly.coordinates + current_beta * diff + alfa * urand_n
        self.coordinates += lambd * np.multiply(urand_n, self.coordinates - gbest_pos)
