import numpy as np
from wolf import Wolf

# Class, which represents a wolf pack.
class Pack:
    # D - dimension of the search space;
    # N - number of wolves to generate;
    # step, l_near, t_max, beta - model parameters;
    # fitness_function - function to optimize;
    # w_range - range of particles' coordinates' values (from -range to range);
    # iter_num - maximum number of iterations.
    def __init__(self, D, N, step, fitness_function, w_range, iter_num, l_near=0, t_max=0, beta=0):
        self.wolves = [Wolf(D, w_range) for _ in range(N)]
        self.dimension = D
        self.particles_num = N
        self.step = step
        self.l_near = l_near
        self.t_max = t_max
        self.beta = beta
        self.fitness_function = fitness_function
        self.p_range = w_range
        self.iter_num = iter_num

        # Choosing random wolf as a lead one.
        rnd_wolf = self.wolves[np.random.randint(N)]
        self.lead_position = rnd_wolf.coordinates
        self.lead_score = fitness_function(rnd_wolf.coordinates)

    # Updates the lead wolf of the pack (is called at every iteration).
    def __choose_lead(self):
        for w in self.wolves:
            curr_score = self.fitness_function(w.coordinates)
            if curr_score < self.lead_score:
                self.lead_score = curr_score
                self.lead_position = np.copy(w.coordinates)
    
    # Update all wolves coordinates
    def __move_all(self):
        for w in self.wolves:
            if np.array_equal(w.coordinates, self.lead_position):
                continue
            w.move(self.lead_position, self.step)

    def optimize(self):
        self.__choose_lead()
        t = 0
        while t < self.iter_num:
            self.__move_all()
            t += 1
            self.__choose_lead()
        return self.lead_position
