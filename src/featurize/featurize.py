import pandas as pd
from featurize import GeneticFeatureGenerator, GeneticFeatureSelector, SymbolicFunction
from typing import Callable, Union
import numpy as np


def featurize(
    X: pd.DataFrame,
    y: pd.Series,
    generate_functions: Union[SymbolicFunction | None] = None,
    generate_fitness_metric: str = "pearson",
    generate_max_generations: int = 30,
    generate_num_features: int = 10,
    generate_population_size: int = 100,
    generate_parsimony_coefficient: float = 0.001,
    generate_tournament_size: int = 3,
    gnerate_crossover_proba: float = 0.8,
    generate_early_termination_iters: int = 10,
    selection_cost_func: Callable = None,
    selection_bigger_is_better: bool = True,
    selection_population_size: int = 100,
    selection_crossover_proba: float = 0.8,
    selection_mutation_proba: float = 0.2,
    selection_max_iters: int = 100,
    selection_early_termination_iters: int = 25,
    n_jobs: int = -1,
    set_seed: int = 8888,
    verbose: bool = False,
) -> pd.DataFrame:
    np.random.seed(set_seed)

    symb = GeneticFeatureGenerator(
        functions=generate_functions,
        fitness=generate_fitness_metric,
        max_generations=generate_max_generations,
        num_features=generate_num_features,
        population_size=generate_population_size,
        tournament_size=generate_tournament_size,
        crossover_prob=gnerate_crossover_proba,
        parsimony_coefficient=generate_parsimony_coefficient,
        early_termination_iters=generate_early_termination_iters,
        verbose=verbose,
        n_jobs=n_jobs,
    )
    symb.fit(X, y)

    new_features = symb.transform(X)
    X_new = pd.concat([X, new_features], axis=1)

    selection = GeneticFeatureSelector(
        cost_func=selection_cost_func,
        bigger_is_better=selection_bigger_is_better,
        population_size=selection_population_size,
        crossover_proba=selection_crossover_proba,
        mutation_proba=selection_mutation_proba,
        max_iters=selection_max_iters,
        early_termination_iters=selection_early_termination_iters,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    _, features = selection.optimize(X_new, y)

    return X_new[features]
