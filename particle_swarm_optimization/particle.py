import numpy as np
import random as rand

# Class, which describes a single particle of a swarm.
class Particle:
    # D - dimension of a search space;
    # p_range - range of coordinates' values (from -range to range).
    def __init__(self, D, p_range=50):
        self.velocity = np.zeros(D)
        self.p_range = p_range
        self.coordinates = np.random.rand(D) * np.power(np.full(D, -1), np.random.randint(2, size = D)) * p_range
        self.best_position = np.copy(self.coordinates)
        self.best_score = float('inf')

    # Update velocity based on particle's and group's best positions.
    def __update_velocity(self, group_best_pos, w, c1, c2):
        self.velocity = w * self.velocity \
            + c1 * rand.random() * (self.best_position - self.coordinates) \
            + c2 * rand.random() * (group_best_pos - self.coordinates) \

    # Update position of a particle.
    def move(self, group_best_pos, w, c1, c2, fitness_function):
        self.__update_velocity(group_best_pos, w, c1, c2)
        self.coordinates += self.velocity
        current_score = fitness_function(self.coordinates)
        if current_score < self.best_score:
            self.best_position = np.copy(self.coordinates)
            self.best_score = current_score

    def __str__(self):
        return (
        'Current velocity = {0};\nPosition: {1};\nBest position so far: {2};\nScore for the best position = {3}.\n' \
        .format(self.velocity, self.coordinates, self.best_position, self.best_score))
