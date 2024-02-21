import numpy as np
import sys
from tqdm import tqdm
import copy


class Individual:
    def __init__(self, genome, mutation_proba=0.1):
        self.genome = np.array(genome)
        self.mutation_proba = mutation_proba
        self.current_cost = sys.maxsize

    def evaluate(self, cost_func):
        self.current_cost = cost_func(self.genome)
        return self.current_cost

    def mutate(self):
        proba = np.random.uniform(size=len(self.genome))
        mask = proba < self.mutation_proba
        self.genome[mask == 1] = 1 - self.genome[mask == 1]

    @classmethod
    def crossover(cls, parent1, parent2, crossover_proba):
        c1, c2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

        if np.random.rand() < crossover_proba:
            pt = np.random.randint(1, len(parent1.genome) - 2)
            c1.genome = np.concatenate([parent1.genome[:pt], parent2.genome[pt:]])
            c2.genome = np.concatenate([parent2.genome[:pt], parent1.genome[pt:]])
        return c1, c2


class BinaryGeneticAlgorithm:
    def __init__(
        self,
        cost_func,
        num_genes,
        population_size=100,
        crossover_proba=0.75,
        mutation_proba=0.1,
        max_iters=150,
        early_termination_iters=10,
        verbose=False,
    ):
        self.cost_func = cost_func
        self.population_size = population_size
        self.num_genes = num_genes
        self.crossover_proba = crossover_proba
        self.mutation_proba = mutation_proba
        self.max_iters = max_iters

        self.early_termination_iters = early_termination_iters
        self.early_termination_counter = 0

        self.history_best = list()
        self.history_mean = list()

        self.best_cost = sys.maxsize
        self.best_genome = None

        self.verbose = verbose

        self.population = [
            Individual(np.random.randint(0, 2, self.num_genes), self.mutation_proba)
            for _ in range(self.population_size)
        ]

    def optimize(self):
        pbar = tqdm(total=self.max_iters, desc="Optimising feature space...")

        for iter in range(self.max_iters):
            scores = []
            for individual in self.population:
                individual.evaluate(self.cost_func)
                scores.append(individual.current_cost)

                if individual.current_cost < self.best_cost:
                    self.best_cost = individual.current_cost
                    self.best_genome = individual.genome
                    self.early_termination_counter = 0
                    if self.verbose:
                        print(f"iter: {iter:>4d}, best error: {self.best_cost:10.6f}")

            self.early_termination_counter += 1
            if self.early_termination_counter >= self.early_termination_iters:
                if self.verbose:
                    print(
                        f"Early termination at iter {iter}, best error: {self.best_cost:10.6f}"
                    )
                break

            self.history_best.append(self.best_cost)
            self.history_mean.append(np.mean(scores))

            selected = [self._selection(scores) for _ in range(self.population_size)]

            children = list()
            for i in range(0, len(self.population) - 1, 2):
                p1, p2 = selected[i], selected[i + 1]
                for c in Individual.crossover(p1, p2, self.crossover_proba):
                    c.mutate()
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
        c1, c2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

        if np.random.rand() < self.crossover_proba:
            pt = np.random.randint(1, len(parent1.genome) - 2)
            c1.genome = np.concatenate([parent1.genome[:pt], parent2.genome[pt:]])
            c2.genome = np.concatenate([parent2.genome[:pt], parent1.genome[pt:]])
        return c1, c2

    # def _adaptive_mutation(self, bitstring, mutation_proba):
    #     proba = np.random.uniform(size=len(bitstring))
    #     mask = proba < mutation_proba
    #     bitstring[mask == 1] = 1 - bitstring[mask == 1]
    #     return bitstring
