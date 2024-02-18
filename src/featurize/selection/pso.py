import numpy as np
from tqdm import tqdm
from typing import Callable
import sys
from featurize.logging import logger


class BaseSwarmOptimizer:
    """
    Base class for swarm optimization algorithms.

    Parameters
    ----------
    num_particles : int
        The number of particles in the swarm.
    num_dimensions : int
        The number of dimensions in the search space.
    """

    def __init__(self, num_particles, num_dimensions):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions

    def optimize(self, cost_function, max_iter, verbose=False):
        """
        Optimize the cost function using the swarm optimization algorithm.

        Parameters
        ----------
        cost_function : function
            The cost function to optimize.

        max_iter : int
            The maximum number of iterations.

        verbose : bool (default=False)
            Whether to print the best error at each iteration.
        """
        raise NotImplementedError


class BaseParticle:
    def __init__(self, num_dimensions):
        """
        Initialize the particle.

        Parameters
        ----------
        num_dimensions : int
            The number of dimensions in the search space.
        """
        self.num_dimensions = num_dimensions
        self.position = np.random.uniform(size=num_dimensions)
        self.velocity = np.random.uniform(size=num_dimensions)
        self.best_position = np.array([])
        self.current_error = sys.maxsize
        self.best_error = sys.maxsize

    def evaluate(self, cost_function, X, y):
        """
        Evaluate the particle's current position.

        Parameters
        ----------
        cost_function : function
            The cost function to optimize.
        X : pd.DataFrame
            The dataframe with the features.
        y : pd.Series
            The target variable.
        """
        raise NotImplementedError

    def update_velocity(self, best_position, current_iter, max_iter):
        """
        Update the particle's velocity.

        Parameters
        ----------
        best_position : np.array
            The best position found by the swarm.
        current_iter : int
            The current iteration.
        max_iter : int
            The maximum number of iterations.
        """
        raise NotImplementedError

    def update_position(self):
        """
        Update the particle's position.
        """
        raise NotImplementedError


class BinaryParticleSwarmOptimiser(BaseSwarmOptimizer):
    """
    Binary Particle Swarm Optimizer.
    """

    def __init__(self, num_particles: int, num_dimensions: int):
        """
        Initialize the Binary Particle Swarm Optimizer.

        Parameters
        ----------
        num_particles : int
            The number of particles in the swarm.
        num_dimensions : int
            The number of dimensions in the search space.
        """
        logger.info("Initialising Binary Particle Swarm Optimiser")
        super().__init__(num_particles, num_dimensions)

        logger.info("Creating swarm of binary particles")
        self.swarm = [BinaryParticle(num_dimensions) for _ in range(self.num_particles)]

    def optimize(self, cost_function: Callable, max_iter: int, verbose=False):
        """
        Optimize the cost function using the Binary Particle Swarm Optimizer.

        Parameters
        ----------
        cost_function : function
            The cost function to optimize.

        max_iter : int
            The maximum number of iterations.

        verbose : bool (default=False)
            Whether to print the best error at each iteration.
        """
        logger.info("Optimising feature space using Binary Particle Swarm Optimiser")
        self.cost_function = cost_function
        self.verbose = verbose
        self.global_best_error = 99999999
        self.global_best_position = np.array([])

        max_ticks = max_iter * self.num_particles
        pbar = tqdm(total=max_ticks, desc="Optimising feature space...")

        for i in range(max_iter):
            if verbose:
                print(f"iter: {i:>4d}, best error: {self.global_best_error:10.6f}")

            for particle in self.swarm:
                particle.evaluate(cost_function)
                if particle.current_error < self.global_best_error:
                    self.global_best_position = particle.position.copy()
                    self.global_best_error = particle.current_error
                pbar.update(1)

            for particle in self.swarm:
                particle.update_velocity(self.global_best_position, i, max_iter)
                particle.update_position()

        return self.global_best_error, self.global_best_position


class BinaryParticle(BaseParticle):
    """
    Binary Particle.
    """

    def __init__(self, num_dimensions):
        """
        Initialize the Binary Particle.

        Parameters
        ----------
        num_dimensions : int
            The number of dimensions in the search space.
        """
        super().__init__(num_dimensions)
        self.position = np.rint(self.position).astype(int)

    def update_velocity(self, best_global_position, current_iter, max_iter):
        """
        Update the particle's velocity.

        Parameters
        ----------
        best_global_position : np.array
            The best position found by the swarm.
        current_iter : int
            The current iteration.
        max_iter : int
            The maximum number of iterations.
        """
        r1 = np.random.random(size=len(self.position))
        r2 = np.random.random(size=len(self.position))

        n = max_iter
        t = current_iter

        w = (0.4 / n**2) * (t - n) ** 2 + 0.4
        c1 = -3 * t / n + 3.5
        c2 = 3 * t / n + 0.5

        vel_cognitive = c1 * r1 * (best_global_position - self.position)
        vel_social = c2 * r2 * (best_global_position - self.position)
        self.velocity = w * self.velocity + vel_cognitive + vel_social

    def update_position(self):
        """
        Update the particle's position.
        """
        self.position = (
            np.random.random_sample(size=self.num_dimensions)
            < self._sigmoid(self.velocity)
        ).astype(int)

    def evaluate(self, cost_func):
        """
        Evaluate the particle's current position.

        Parameters
        ----------
        cost_func : function
            The cost function to optimize.
        """
        self.current_error = cost_func(self.position)

        if self.current_error < self.best_error:
            self.best_position = self.position.copy()
            self.best_error = self.current_error

    @staticmethod
    def _sigmoid(x):
        """
        Sigmoid function.

        Parameters
        ----------
        x : np.array
            The input array.
        """
        x = np.array(x)
        return 1 / (1 + np.exp(-x))
