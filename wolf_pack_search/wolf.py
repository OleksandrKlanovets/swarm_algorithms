import numpy as np
import random as rand

# Class, which represents a single wolf of a pack.
class Wolf:
    # D - dimension of a search space;
    # w_range - range of coordinates' values (from -range to range).
    def __init__(self, D, w_range):
        self.dimension = D
        self.w_range = w_range
        self.coordinates = np.random.rand(D) * w_range
        self.coordinates *= np.power(np.full(D, -1), np.random.randint(2, size = D))

    def move(self, group_best_pos, step):
        # print('----------------------------------')
        # print('G_Best: {0}'.format(group_best_pos))
        # print('Coords: {0}'.format(self.coordinates))
        diff = group_best_pos - self.coordinates
        # print('Diff: {0}'.format(diff))
        norm = np.linalg.norm(diff)
        # print('Norm: {0}'.format(norm))
        self.coordinates += step * diff / norm
        # print('Modified coords: {0}'.format(self.coordinates))
