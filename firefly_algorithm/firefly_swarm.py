import numpy as np
from firefly import Firefly


# Class, which represents a swarm of fireflies.
class FireflySwarm:
    # D - dimension of the search space;
    # N - number of wolves to generate;
    # beta_0 - attractiveness of fireflies at 0 distance;
    # alfa_0, alfa_inf - initial and final value of alfa parameter respectively;
    # gamma - light absorption coefficient;
    # fit_func - target function to optimize;
    # iter_num - maximum number of iterations;
    # p_range - range of fireflies coordinates values (from -range to range);
    def __init__(self, D, N, beta_0, alfa_0, alfa_inf, gamma, lambd, fit_func, iter_num, p_range):
        self.D = D
        self.N = N
        self.beta_0 = beta_0
        self.alfa_0 = alfa_0
        self.alfa_inf = alfa_inf
        self.gamma = gamma
        self.lambd = lambd
        self.fit_func = fit_func
        self.iter_num = iter_num
        self.p_range = p_range
        self.fireflies = [Firefly(D, p_range, fit_func) for _ in range(N)]

        # Setting random firefly as group's initial best position (we'll see if we need that).
        rnd_firefly = self.fireflies[np.random.randint(N)]
        self.group_best_position = np.copy(rnd_firefly.coordinates)
        self.group_best_score = fit_func(rnd_firefly.coordinates)

    # Updates best group score (is called at each iteration).
    def __update_group_best(self):
        for f in self.fireflies:
            curr_score = self.fit_func(f.coordinates)
            if curr_score < self.group_best_score:
                self.group_best_score = curr_score
                self.group_best_position = np.copy(f.coordinates)

    # Update position of all the fireflies in the swarm.
    def __move_all(self, alfa):
        for i in range(self.N):
            for j in range(i):
                if self.fireflies[j].score < self.fireflies[i].score:
                    # Move less attractive to more attractive.
                    self.fireflies[i].move_towards_firefly(
                        self.fireflies[j],
                        self.beta_0,
                        alfa,
                        self.gamma,
                        self.lambd,
                        self.group_best_position,
                        self.fit_func
                    )

    # Returns a key for fireflies' ranking.
    def __score_key(self, val):
        return val.score

    def optimize(self):
        t = 0
        alfa = self.alfa_0
        while(t < self.iter_num):
            # Loop through all fireflies pairs.
            self.__move_all(alfa)

            # Rank the fireflies.
            self.fireflies.sort(key=self.__score_key)

            # Find current best.
            self.__update_group_best()
            
            # Update alfa.
            alfa = self.alfa_inf + (self.alfa_0 - self.alfa_inf) * np.exp(-t)
            t += 1
        return self.group_best_position
