import numpy as np


class BaseSwarmOptimizer:
    def __init__(self, num_particles, num_dimensions, options):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.options = options

    def optimize(self, cost_function, bounds, max_iter, verbose=False):
        raise NotImplementedError


class BaseParticle:
    def __init__(self, num_dimensions, options):
        self.num_dimensions = num_dimensions
        self.position = np.random.uniform(size=num_dimensions)
        self.velocity = np.random.uniform(size=num_dimensions)
        self.best_position = np.array([])
        self.current_error = 9999999
        self.best_error = 99999999
        self.options = options

    def evaluate(self, cost_function, X, y):
        raise NotImplementedError

    def update_velocity(self, best_position, current_iter, max_iter):
        raise NotImplementedError

    def update_position(self, bounds):
        raise NotImplementedError


class BinaryParticleSwarmOptimiser(BaseSwarmOptimizer):
    def __init__(self, num_particles, num_dimensions, options):
        super().__init__(num_particles, num_dimensions, options)

        self.swarm = [
            BinaryParticle(num_dimensions, options) for _ in range(self.num_particles)
        ]

    def optimize(self, cost_function, max_iter, verbose=False):
        self.cost_function = cost_function
        self.verbose = verbose
        self.global_best_error = 99999999
        self.global_best_position = np.array([])

        for i in range(max_iter):
            if verbose:
                print(f"iter: {i:>4d}, best error: {self.global_best_error:10.6f}")

            for particle in self.swarm:
                particle.evaluate(cost_function)
                if particle.current_error < self.global_best_error:
                    self.global_best_position = particle.position.copy()
                    self.global_best_error = particle.current_error

            for particle in self.swarm:
                particle.update_velocity(self.global_best_position, i, max_iter)
                particle.update_position()

        return self.global_best_error, self.global_best_position


class BinaryParticle(BaseParticle):
    def __init__(self, num_dimensions, options):
        super().__init__(num_dimensions, options)
        self.position = np.rint(self.position).astype(int)

    def update_velocity(self, best_global_position, current_iter, max_iter):
        r1 = np.random.random(size=len(self.position))
        r2 = np.random.random(size=len(self.position))

        # c1 = self.options["c1"]
        # c2 = self.options["c2"]
        # w = self.options["w"]

        n = max_iter
        t = current_iter

        w = (0.4 / n**2) * (t - n) ** 2 + 0.4
        c1 = -3 * t / n + 3.5
        c2 = 3 * t / n + 0.5

        vel_cognitive = c1 * r1 * (best_global_position - self.position)
        vel_social = c2 * r2 * (best_global_position - self.position)
        self.velocity = w * self.velocity + vel_cognitive + vel_social

    def update_position(self):
        self.position = (
            np.random.random_sample(size=self.num_dimensions)
            < self._sigmoid(self.velocity)
        ).astype(int)

    def evaluate(self, cost_func):
        self.current_error = cost_func(self.position)

        # check to see if the current position is an individual best
        if self.current_error < self.best_error:
            self.best_position = self.position.copy()
            self.best_error = self.current_error

    def _sigmoid(self, x):
        x = np.array(x)
        return 1 / (1 + np.exp(-x))


# class Particle:
#     def __init__(self, x0):
#         self.position = x0  # particle starting positions
#         self.velocity = np.random.uniform(size=len(x0))  # particle starting velocities
#         self.pos_best_i = np.array([])  # best position individual
#         self.err_best_i = -1  # best error individual
#         self.err_i = -1  # error individual

#     # evaluate current fitness
#     def evaluate(self, costFunc, X, y):
#         x0 = np.rint(self.position).astype(int)
#         cols = X.columns[x0 == 1]
#         self.err_i = costFunc(cols, X, y)

#         # check to see if the current position is an individual best
#         if self.err_i < self.err_best_i or self.err_best_i == -1:
#             self.pos_best_i = self.position.copy()
#             self.err_best_i = self.err_i

#     # update new particle velocity
#     def update_velocity(self, pos_best_g, current_iter, max_iter):
#         w = 0.8  # constant inertia weight (how much to weigh the previous velocity)
#         c1 = 0.5  # cognative constant
#         c2 = 0.5  # social constant

#         # w_min = 0.5
#         # w_max = 2
#         # w = w_max - ((w_max - w_min) / max_iter) * current_iter

#         r1 = np.random.random(size=len(self.position))
#         r2 = np.random.random(size=len(self.position))

#         vel_cognitive = c1 * r1 * (self.pos_best_i - self.position)
#         vel_social = c2 * r2 * (pos_best_g - self.position)
#         self.velocity = w * self.velocity + vel_cognitive + vel_social

#     # update the particle position based off new velocity updates
#     def update_position(self, bounds):
#         self.position = self.position + self.velocity
#         self.position = np.clip(self.position, bounds[0], bounds[1])


# def pso(costFunc, X, y, bounds, num_particles, maxiter, verbose=False):
#     num_dimensions = len(X.columns)
#     err_best_g = -1  # best error for group
#     pos_best_g = []  # best position for group

#     # establish the swarm
#     swarm = []
#     for i in range(0, num_particles):
#         x0 = np.random.randint(2, size=num_dimensions)
#         swarm.append(Particle(x0))

#     # begin optimization loop
#     i = 0
#     while i < maxiter:
#         if verbose:
#             print(f"iter: {i:>4d}, best solution: {err_best_g:10.6f}")

#         # cycle through particles in swarm and evaluate fitness
#         for j in range(0, num_particles):
#             swarm[j].evaluate(costFunc, X, y)

#             # determine if current particle is the best (globally)
#             if swarm[j].err_i < err_best_g or err_best_g == -1:
#                 pos_best_g = list(swarm[j].position)
#                 err_best_g = float(swarm[j].err_i)

#         # cycle through swarm and update velocities and position
#         for j in range(0, num_particles):
#             swarm[j].update_velocity(pos_best_g, i, maxiter)
#             swarm[j].update_position(bounds)
#         i += 1

#     x0 = np.rint(pos_best_g).astype(int)

#     # print final results
#     if verbose:
#         print(f"   > {pos_best_g}")
#         print(f"   > {err_best_g}\n")
#         print(f"   > {x0.sum()}\n")

#     return X.columns[x0 == 1], pos_best_g
