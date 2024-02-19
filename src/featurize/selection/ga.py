import numpy as np
import sys
from tqdm import tqdm


class GeneticAlgorithm:
    def __init__(
        self,
        cost_func,
        num_features,
        num_individuals=100,
        crossover_proba=0.5,
        mutation_proba=0.1,
        max_iters=150,
    ):
        self.cost_func = cost_func
        self.num_individuals = num_individuals
        self.num_features = num_features
        self.crossover_proba = crossover_proba
        self.mutation_proba = mutation_proba
        self.max_iters = max_iters

        self.best_cost = sys.maxsize
        self.best_genome = None

        self.population = [
            np.random.randint(0, 2, num_features).tolist()
            for _ in range(num_individuals)
        ]

    def optimize(self):
        pbar = tqdm(total=self.max_iters, desc="Optimising feature space...")

        for iter in range(self.max_iters):
            scores = [self.cost_func(np.array(c)) for c in self.population]

            for i in range(len(self.population)):
                if scores[i] < self.best_cost:
                    self.best_cost = scores[i]
                    self.best_genome = self.population[i]
                    print(f"iter: {iter:>4d}, best error: {self.best_cost:10.6f}")

            selected = [self._selection(scores) for _ in range(self.num_individuals)]

            children = list()
            for i in range(0, len(self.population) - 1, 2):
                p1, p2 = selected[i], selected[i + 1]
                for c in self._crossover(p1, p2):
                    c = self._mutation(c)
                    children.append(c)

            # replace population
            self.population = children
            pbar.update(1)
        return self.best_cost, np.array(self.best_genome)

    def _selection(self, scores, k=3):
        selection_ix = np.random.randint(len(self.population))

        for ix in np.random.randint(0, len(self.population), k - 1):
            if scores[ix] < scores[selection_ix]:
                selection_ix = ix
        return self.population[selection_ix]

    def _crossover(self, parent1, parent2):
        c1, c2 = parent1.copy(), parent2.copy()

        if np.random.rand() < self.crossover_proba:
            pt = np.random.randint(1, len(parent1) - 2)
            c1 = parent1[:pt] + parent2[pt:]
            c2 = parent2[:pt] + parent1[pt:]
        return c1, c2

    def _mutation(self, bitstring):
        for i in range(len(bitstring)):
            if np.random.rand() < self.mutation_proba:
                bitstring[i] = 1 - bitstring[i]
        return bitstring
