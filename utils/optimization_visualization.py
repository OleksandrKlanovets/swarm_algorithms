import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_function(func, l_bound, u_bound, step):
    '''
    Plots function colormap.

    Parameters
    ----------
    func : callable
        Function to plot.
    l_bound : float
        Lower bound for X and Y axes.
    u_bound : float
        Upper bound for X and Y axes.
    step : float
        Spacing between points, which is used to calculate 
        fitness-function value for each particle.

    Returns
    -------
    No value.
    '''
    # Create a grid for displaying the search space.
    X = np.arange(l_bound, u_bound, step)
    Y = np.copy(X)

    # Get the values of the target function at cartesian_square(X, Y).
    Z = []
    for i in range(Y.__len__()):
        temp = []
        for j in range(X.__len__()):
            temp.append(func(np.array([X[j], Y[i]])))
        Z.append(temp)

    # Plot the target function colormap.
    plt.contourf(X, Y, Z, 100)


def plot_particles(model):
    '''
    Plots the particles at their current positions.

    Parameters
    ----------
    model : SwarmAlgorithm
        An initialized model used to perform optimization.

    Returns
    -------
    plotted_particles : matplotlib.lines.Line2D
        Plotted swarm without the global best.
    gbest : matplotlib.lines.Line2D
        Plotted global best particle.
    '''
    indices = np.arange(model.N)
    gbest_index = np.argmin(model.scores)

    X = model.particles[indices != gbest_index, 0]
    Y = model.particles[indices != gbest_index, 1]

    X_gbest = model.particles[gbest_index, 0]
    Y_gbest = model.particles[gbest_index, 1]

    NON_GBEST_SIZE = 8
    GBEST_SIZE = 12
    plotted_particles, = plt.plot(X, Y, marker='X', color='r', 
                                  linewidth=0, markersize=NON_GBEST_SIZE)
    gbest, = plt.plot(X_gbest, Y_gbest, marker='*', color='y',
                      linewidth=0, markersize=GBEST_SIZE)
    return plotted_particles, gbest


def animate(i, model, plotted_particles):
    '''
    Refreshes the particles.

    Parameters
    ----------
    i : int

    model : SwarmAlgorithm
        An initialized model used to perform optimization.
    plotted_particles : matplotlib.lines.Line2D
        Plotted particles to update.
    Returns
    -------
    plotted_particles : matplotlib.lines.Line2D
        Updated plotted particles.
    '''
    if i > 1:
        model.optimize()
    
    indices = np.arange(model.N)
    gbest_index = np.argmin(model.scores)
    gbest = model.particles[gbest_index]
    non_gbest = model.particles[indices != gbest_index]

    # Set new positions for global best and the rest of the particles.
    plotted_particles[0].set_data(non_gbest[:, 0], non_gbest[:, 1])
    plotted_particles[1].set_data(gbest[0], gbest[1])
    return plotted_particles,


def visualize_optimization(model, frames_num=100, iteration_duration=200):
    '''
    Visualizes the optimization process for single-objective problems
    with 2 parameters.

    Parameters
    ----------
    model : SwarmAlgorithm
        An initialized single-objective model used to perform optimization.
    frames_num : int, optional, default=100
        Number of frames of the animation (number of generations to visualize).
    i_duration : int, optional, default=200
        Interval/frame duration in milliseconds.

    Returns
    -------
    No value.
    '''
    model.max_iter = 1

    # Create a figure to visualize on.
    fig = plt.figure()

    # Define bounds.
    l_bound = np.min(model.l_bounds)
    u_bound = np.max(model.u_bounds)
    
    FRACTION_DEGREE = 100
    step = (u_bound - l_bound) / FRACTION_DEGREE

    model_name = type(model).__name__
    fit_func_name = model.fit_func.__name__.capitalize()

    plot_function(model.fit_func, l_bound - step, u_bound + step, step)

    # Plot legend.
    plt.xlabel('$X_1$')
    plt.ylabel('$X_2$')
    plt.title(model_name + ' visualization for ' + fit_func_name)
    plt.colorbar()

    # Plot particles at their initial position.
    plotted_particles = plot_particles(model)

    animation = FuncAnimation(
        fig, 
        animate, 
        frames=frames_num, 
        interval=iteration_duration,
        fargs=(model, plotted_particles), 
        repeat=False
    )
    return animation
