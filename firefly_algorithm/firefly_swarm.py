import numpy as np
from firefly import Firefly


# Class, which represents a swarm of fireflies.
class FireflySwarm:
    # D - dimension of the search space;
    # N - number of wolves to generate;
    # beta_0 - attractiveness of fireflies at 0 distance;
    # alfa_0, alfa_inf - initial and final value of alfa parameter respectively;
    # gamma - light absorption coefficient;
    # fitness_func - target function to optimize;
    # iter_num - maximum number of iterations;
    # p_range - range of fireflies coordinates values (from -range to range);
    def __init__(self, D, N, beta_0, alfa_0, alfa_inf, gamma, lambd, fitness_func, iter_num, p_range):
        self.D = D
        self.N = N
        self.beta_0 = beta_0
        self.alfa_0 = alfa_0
        self.alfa_inf = alfa_inf
        self.gamma = gamma
        self.lambd = lambd
        self.fitness_func = fitness_func
        self.iter_num = iter_num
        self.p_range = p_range
        self.fireflies = [Firefly(D, p_range) for _ in range(N)]

        # Setting random firefly as group's initial best position (we'll see if we need that).
        rnd_firefly = self.fireflies[np.random.randint(N)]
        self.group_best_position = np.copy(rnd_firefly.coordinates)
        self.group_best_score = fitness_func(rnd_firefly.coordinates)

    # Updates best group score (is called at each iteration).
    def __update_group_best(self):
        for f in self.fireflies:
            curr_score = self.fitness_func(f.coordinates)
            if curr_score < self.group_best_score:
                self.group_best_score = curr_score
                self.group_best_position = np.copy(f.coordinates)

    # Gets fitness function values vector for all the fireflies.
    def __get_fitness(self):
        result_vec = []
        for f in self.fireflies:
            result_vec.append(self.fitness_func(f.coordinates))
        return result_vec

    def optimize(self):
        t = 0
        alfa = self.alfa_0
        # fitness = self.__get_fitness()
        while(t < self.iter_num):
            # Loop through all fireflies pairs.
            for i in range(self.N):
                for j in range(0, i):
                    f_i = self.fireflies[i].coordinates
                    f_j = self.fireflies[j].coordinates
                    # Compare their attractiveness.
                    # if fitness[j] < fitness[i]:
                    if self.fitness_func(f_j) < self.fitness_func(f_i):
                        # Move less attractive to more attractive.
                        self.fireflies[i].move_towards_firefly(
                            self.fireflies[j],
                            self.beta_0,
                            alfa,
                            self.gamma,
                            self.lambd,
                            self.group_best_position
                        )
            alfa = self.alfa_inf + (self.alfa_0 - self.alfa_inf) * np.exp(-t)
            # fitness = self.__get_fitness()
            t += 1
            self.__update_group_best()
        return self.group_best_position
