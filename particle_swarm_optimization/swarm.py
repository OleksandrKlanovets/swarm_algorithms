import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import particle

# Class, which describes our model, it's states of the optimization process.
# Also contains methods for visualizing how the algorithm works.
class Swarm:
    # D - dimension of the search space;
    # N - number of particles to generate;
    # w, c1, c2 - model parameters;
    # fitness_function - function to optimize;
    # p_range - range of particles' coordinates' values (from -range to range);
    # iter_num - maximum number of iterations.
    def __init__(self, D, N, w, c1, c2, fitness_function, p_range, iter_num):
        self.particles = [particle.Particle(D, p_range) for _ in range(N)]
        self.dimension = D
        self.particles_num = N
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.fitness_function = fitness_function
        self.p_range = p_range
        self.iter_num = iter_num
        self.group_best_score = float('inf')
        rnd_point = np.random.rand(D) * np.power(np.full(D, -1), np.random.randint(2, size = D)) * p_range
        self.group_best_position = rnd_point

    # Updates best group position and score (is called at every iteration).
    def __update_group_best(self):
        for p in self.particles:
            if p.best_score < self.group_best_score:
                self.group_best_score = p.best_score
                self.group_best_position = np.copy(p.coordinates)

    # Updates the positions of all the particles in the swarm.
    def __move_all(self):
        for p in self.particles:
            p.move(self.group_best_position, self.w, self.c1, self.c2, self.fitness_function)

    # Main loop of the algorithm.
    def optimize(self):
        i = 0
        self.__update_group_best()
        while(i < self.iter_num):
            self.__move_all()
            self.__update_group_best()
            i += 1

    # Plots the particles at their current position.
    def __plot_particles(self):
        X = []
        Y = []
        for p in self.particles:
            X.append(p.coordinates[0])
            Y.append(p.coordinates[1])
        plotted_particles, = plt.plot(X, Y, marker='x', color='r', linewidth=0)
        return plotted_particles
    
    # Refreshes the particles.
    def __animate(self, i, plotted_particles):
        self.__move_all()
        self.__update_group_best()
        X = []
        Y = []
        for p in self.particles:
            X.append(p.coordinates[0])
            Y.append(p.coordinates[1])
        plotted_particles.set_data(X, Y)
        return plotted_particles,

    # Visualizes the optimization process. Perhaps, not the best implementation ever...
    # xmin, xmax, ymin, ymax - boundaries of the displayed search space.
    # step - spacing between points, which are used to calculate fitness function.
    # fnum - number of frames of the animation.
    # i_duration - interval duration.
    def visualize(self, step, f_num, i_duration):
        if self.dimension != 2:
            err_msg = 'ERROR: The search space of dimension D={0} is impossible to visualize.'
            print(err_msg.format(self.dimension))
            return
        # Create a figure to visualize on.
        fig = plt.figure()

        # Create a grid for displaying the search space.
        X = np.arange(-self.p_range, self.p_range, step)
        Y = np.arange(-self.p_range, self.p_range, step)
        # XX, YY = np.meshgrid(X, Y)

        # Get the values of the target function at cartesian_square(X, Y).
        Z = []
        for i in range(X.__len__()):
            temp = []
            for j in range(Y.__len__()):
                temp.append(self.fitness_function(np.array([X[i], Y[j]])))
            Z.append(temp)

        # Plot the target function colormap.
        plt.contourf(X, Y, Z, 50)

        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('PSO algorithm visualization')
        plt.colorbar()

        # Plot particles at their initial position.
        plotted_particles = self.__plot_particles()
        
        # Initial update.
        self.__update_group_best()

        # Create the animation itself
        animation = FuncAnimation(fig, self.__animate, frames=f_num, interval=i_duration, 
                                  fargs=(plotted_particles,), blit=True, repeat=False)

        plt.show()


    def __str__(self):
        return 'Best one:{0}'.format(self.group_best_position)
