"""Contains the GeneticFeatureSynthesis class."""

from typing import Callable, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import cpu_count
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import type_of_target
from tqdm.auto import tqdm

from featuristic.core.models import ProgramFitness
from featuristic.core.mrmr import MaxRelevanceMinRedundancy
from featuristic.core.preprocess import preprocess_data
from featuristic.core.program import node_count, render_prog, simplify_prog_str
from featuristic.core.registry import FUNCTION_REGISTRY, SymbolicFunction
from featuristic.core.symbolic_population import SymbolicPopulation
from featuristic.fitness.registry import get_fitness
from featuristic.core.optimizer import optimize_constants

PARSINOMY_STRENGTH = 3.0


class GeneticFeatureSynthesis(BaseEstimator, TransformerMixin):
    """
    The GeneticFeatureSynthesis class uses genetic programming to generate new
    features using a technique based on Symbolic Regression. This is done by initially
    building a population of naive random formulas that represent transformations of
    the input features. The population is then evolved over a number of generations
    using genetic functions such as mutation and crossover to find the best programs
    that minimize a given fitness function. The best features are then identified using
    a Maximum Relevance Minimum Redundancy (mRMR) algorithm to find those features
    that are most correlated with the target variable while being least correlated with
    each other.
    """

    def __init__(
        self,
        num_features: int = 10,
        population_size: int = 50,
        max_generations: int = 25,
        tournament_size: int = 7,
        crossover_proba: float = 0.75,
        parsimony_coefficient: float = 0.001,
        adaptive_parsimony: bool = True,
        early_termination_iters: int = 15,
        functions: Optional[Union[List[str], List[SymbolicFunction]]] = None,
        custom_functions: Optional[List[SymbolicFunction]] = None,
        fitness_function: Optional[Union[str, Callable]] = None,
        return_all_features: bool = True,
        n_jobs: int = -1,
        show_progress_bar: bool = True,
        verbose: bool = False,
        min_constant_val: float = -10.0,
        max_constant_val: float = 10.0,
        include_constants: bool = True,
        optimize_constants: bool = True,
        constant_optimization_maxiter: int = 100,
        const_prob: float = 0.15,
        stop_prob: float = 0.8,
        max_depth: int = 3,
    ):
        """
        Initialize the GeneticFeatureSynthesis.

        Args
        ----
        num_features : int
            The number of best features to generate. Internally, `3 * num_features`
            programs are generated and the
            best `num_features` are selected via Maximum Relevance Minimum Redundancy
            (mRMR).

        population_size : int
            The number of programs in each generation. The larger the population, the
            more likely it is to find a good solution, but the longer it will take.

        max_generations : int
            The maximum number of generations to run. The larger the number of
            generations, the more likely it is to find a good solution, but the longer
            it will take.

        tournament_size : int
            The size of the tournament for selection. The larger the tournament size,
            the more likely it is to select the best program, but the more computation
            it will take.

        crossover_proba : float
            The probability of crossover mutation between selected parents in each
            generation.

        parsimony_coefficient : float
            The parsimony coefficient. Larger values penalize larger programs more and
            encourage smaller programs. This helps prevent bloat where the programs
            become increasingly large and complex without improving the fitness, which
            increases computation complexity and reduces the interpretability of the
            features.

        early_termination_iters : int
            If the best score does not improve for this number of generations, then the
            algorithm will terminate early.

        functions : list[str] | list[SymbolicFunction]
            The list of functions to use in the programs. If `None` then all the
            built-in functions are used. The functions must be the names of the
            functions returned by the `list_symbolic_functions` method.

        custom_functions : list[SymbolicFunction]
            A list of custom functions to use in the programs. Each custom function
            must be an instance of the `CustomSymbolicFunction` class.

        fitness_function : Callable | None
            The fitness function to use to evaluate the programs. If `None`, then the
            fitness function will be inferred based on the target type.

        return_all_features : bool
            Whether to return all the features generated or just the best features.

        n_jobs : int
            The number of parallel jobs to run. If `-1`, use all available cores else
            uses n_jobs. If `n_jobs=1`, then the computation is done in serial.

        show_progress_bar: bool
            Whether to show a progress bar.

        verbose : bool
            Whether to print out aditional information

        min_constant_val : float
            The minimum value for ephemeral random constants generated during
            program creation. Default is -10.0.

        max_constant_val : float
            The maximum value for ephemeral random constants generated during
            program creation. Default is 10.0.

        include_constants : bool
            Whether to include ephemeral random constants in the generated programs.
            If False, programs will only use input features and functions without
            any constant values. Default is True.

        optimize_constants : bool
            Whether to optimize the constants in the generated programs. Default is True.

        constant_optimization_maxiter : int
            The maximum number of iterations for the constant optimization. Default is 100.

        const_prob : float
            The probability of generating a constant leaf node.

        stop_prob : float
            The probability of stopping the program generation.

        max_depth : int
            The maximum depth of the programs.
        """
        if functions is None:
            self.functions = list(FUNCTION_REGISTRY.values())
        else:
            self.functions = []
            for name in functions:
                if name not in FUNCTION_REGISTRY:
                    raise ValueError(
                        f"Function '{name}' not found in FUNCTION_REGISTRY."
                    )
                self.functions.append(FUNCTION_REGISTRY[name])

        if custom_functions is not None:
            self.functions.extend(custom_functions)

        self.population_size = population_size
        self.max_generations = max_generations
        self.tournament_size = tournament_size
        self.crossover_proba = crossover_proba
        self.num_features = num_features

        self.parsimony_coefficient = parsimony_coefficient
        self._base_parsimony = parsimony_coefficient
        self._initial_avg_size = None
        self.adaptive_parsimony = adaptive_parsimony

        self.history = []
        self.hall_of_fame = []
        self.len_hall_of_fame = self.num_features * 5

        self.population = None
        self.early_termination_iters = early_termination_iters
        self.early_termination_counter = 0

        self.return_all_features = return_all_features

        if isinstance(fitness_function, str):
            self.fitness_function = get_fitness(fitness_function)
        elif callable(fitness_function):
            self.fitness_function = fitness_function
        else:
            self.fitness_function = None

        self.verbose = verbose

        self.fit_called = False

        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs

        self.show_progress_bar = show_progress_bar

        if min_constant_val > max_constant_val:
            raise ValueError(
                "min_constant_val must be less than or equal to max_constant_val."
            )
        self.min_constant_val = min_constant_val
        self.max_constant_val = max_constant_val
        self.include_constants = include_constants

        self.optimize_constants = optimize_constants
        self.constant_optimization_maxiter = constant_optimization_maxiter

        self.const_prob = const_prob
        self.stop_prob = stop_prob
        self.max_depth = max_depth

    def _update_hall_of_fame(self, fitness: List[float]):
        for individual, fit in zip(self.population.population, fitness):
            current_fitnesses = [x.fitness for x in self.hall_of_fame]
            if fit not in current_fitnesses:
                self.hall_of_fame.append(
                    ProgramFitness(fitness=fit, individual=individual)
                )

        self.hall_of_fame.sort()
        self.hall_of_fame = self.hall_of_fame[: self.len_hall_of_fame]

    def _select_best_features(self, X: pd.DataFrame, y: pd.Series):
        """
        Select the best features using the mRMR algorithm.

        Args
        ----
        X : pd.DataFrame
            The dataframe with the features.

        y : pd.Series
            The target variable.

        return
        ------
        None
        """
        if not self.hall_of_fame:
            return

        population = SymbolicPopulation(
            len(self.hall_of_fame),
            self.functions,
            self.tournament_size,
            self.crossover_proba,
            n_jobs=1,  # Use serial processing for feature selection
            min_constant_val=self.min_constant_val,
            max_constant_val=self.max_constant_val,
            include_constants=self.include_constants,
            const_prob=self.const_prob,
            stop_prob=self.stop_prob,
            max_depth=self.max_depth,
        )

        population.population = [x.individual for x in self.hall_of_fame]
        features = pd.DataFrame(population.evaluate(X)).T

        num_evaluated_features = features.shape[1]
        features.columns = [f"feature_{i}" for i in range(num_evaluated_features)]

        # Filter hall of fame to only include successfully evaluated features
        successfully_evaluated_indices = features.columns.str.replace(
            "feature_", ""
        ).astype(int)
        self.hall_of_fame = [
            self.hall_of_fame[i] for i in successfully_evaluated_indices
        ]

        for i in range(len(self.hall_of_fame)):
            self.hall_of_fame[i].name = f"feature_{i}"

        selected = (
            MaxRelevanceMinRedundancy(
                k=self.num_features, show_progress_bar=self.show_progress_bar
            )
            .fit_transform(features, y)
            .columns
        )
        selected = [int(x.split("_")[1]) for x in selected]
        self.hall_of_fame = [self.hall_of_fame[i] for i in selected]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GeneticFeatureSynthesis":
        """
        Fit the symbolic feature generator to the data.

        Args
        ----
        X : pd.DataFrame
            The dataframe with the features.

        y : pd.Series
            The target variable.

        return
        ------
        returns self
        """
        if X is None or y is None or X.empty or y.empty:
            raise ValueError("Input features and target must not be empty.")
        if X.isnull().all().all() or y.isnull().all():
            raise ValueError("Input features and target must not be all NaN.")
        X_copy, y_copy = preprocess_data(
            X.reset_index(drop=True), y.reset_index(drop=True)
        )

        if self.fitness_function is None:
            target_type = type_of_target(y_copy)
            if target_type in ("binary", "multiclass"):
                self.fitness_function = get_fitness("log_loss")
            elif target_type in ("continuous", "continuous-multioutput"):
                self.fitness_function = get_fitness("pearson")
            else:
                raise ValueError(f"Unsupported target type: {target_type}")

        # Initialize the population
        self.population = SymbolicPopulation(
            self.population_size,
            self.functions,
            self.tournament_size,
            self.crossover_proba,
            n_jobs=self.n_jobs,
            min_constant_val=self.min_constant_val,
            max_constant_val=self.max_constant_val,
            include_constants=self.include_constants,
            const_prob=self.const_prob,
            stop_prob=self.stop_prob,
            max_depth=self.max_depth,
        ).initialize(X_copy)

        # loss value to minimize
        global_best = float("inf")
        best_prog = self.population.population[
            0
        ]  # Initialize to first program to avoid None

        pbar = None
        if self.show_progress_bar and self.n_jobs == 1:
            pbar = tqdm(total=self.max_generations, desc="Creating new features...")

        for gen in range(self.max_generations):
            fitness = []
            prediction = self.population.evaluate(X_copy)

            if self._initial_avg_size is None:
                self._initial_avg_size = np.mean(
                    [node_count(p) for p in self.population.population]
                )

            if self.adaptive_parsimony and self._initial_avg_size > 0:
                avg_size = np.mean([node_count(p) for p in self.population.population])
                self.parsimony_coefficient = (
                    self._base_parsimony
                    * (avg_size / self._initial_avg_size)
                    * PARSINOMY_STRENGTH
                )

            score = self.population.compute_fitness(
                self.fitness_function, self.parsimony_coefficient, prediction, y_copy
            )

            top_n = 100
            top_indices = np.argsort(score)[:top_n]
            for idx in top_indices:
                prog = self.population.population[idx]
                loss_fn = lambda y_true, y_pred: self.fitness_function(
                    prog, self.parsimony_coefficient, y_true, y_pred
                )
                self.population.population[idx] = optimize_constants(
                    prog,
                    X,
                    y,
                    loss_fn=loss_fn,
                    maxiter=self.constant_optimization_maxiter,
                    min_val=self.min_constant_val,
                    max_val=self.max_constant_val,
                )

            prediction = self.population.evaluate(X_copy)
            score = self.population.compute_fitness(
                self.fitness_function, self.parsimony_coefficient, prediction, y_copy
            )

            improved = False
            for prog, s in zip(self.population.population, score):
                fitness.append(s)
                if s < global_best:
                    global_best = s
                    best_prog = prog
                    improved = True

            # update the history
            results = {
                "generation": gen,
                "best_score": global_best,
                "median_score": pd.Series(fitness).median(),
                "best_program": render_prog(best_prog),
                "parsimony": self.parsimony_coefficient,
            }
            self.history.append(results)

            if improved:
                self.early_termination_counter = 0
            else:
                self.early_termination_counter += 1

            # check for early termination
            if self.early_termination_counter >= self.early_termination_iters:
                if self.verbose:
                    print(
                        f"Early termination at generation {gen}, "
                        f"no improvement for {self.early_termination_iters} generations. "
                        f"Best error: {global_best:10.6f}"
                    )
                break

            if pbar:
                pbar.update(1)

            self._update_hall_of_fame(fitness)

            self.population.evolve(fitness, X_copy)

        if pbar:
            pbar.close()

        self._select_best_features(X_copy, y_copy)

        if self.verbose:
            print("Symbolic Feature Generator")
            print(f"Best program: {render_prog(best_prog)}")
            print(f"Best score: {global_best}")

        self.fit_called = True

        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Transform the dataframe of features using the best programs found.

        Args
        ----
        X : pd.DataFrame
            The dataframe with the features.

        return
        ------
        pd.DataFrame
            The transformed dataframe.
        """
        if not self.fit_called:
            raise ValueError("Must call fit before transform")

        population = SymbolicPopulation(
            len(self.hall_of_fame),
            self.functions,
            self.tournament_size,
            self.crossover_proba,
            n_jobs=self.n_jobs,
            min_constant_val=self.min_constant_val,
            max_constant_val=self.max_constant_val,
            include_constants=self.include_constants,
            const_prob=self.const_prob,
            stop_prob=self.stop_prob,
            max_depth=self.max_depth,
        )

        population.population = [x.individual for x in self.hall_of_fame]
        output = pd.DataFrame(population.evaluate(X.reset_index(drop=True))).T
        output.columns = [x.name for x in self.hall_of_fame]

        if self.return_all_features:
            return pd.concat([X.reset_index(drop=True), output], axis=1)

        return output

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fit the symbolic feature generator to the data and transform the dataframe of features.

        Args
        ----
        X : pd.DataFrame
            The dataframe with the features.

        y : pd.Series
            The target variable.

        return
        ------
        pd.DataFrame
            The transformed dataframe.
        """
        self.fit(X, y)
        return self.transform(X, y)

    def get_feature_info(self, simplify: bool = True) -> pd.DataFrame:
        """
        Get the information about the best programs found.

        return
        ------
        pd.DataFrame
            The dataframe with the information.
        """
        if not self.fit_called:
            raise ValueError("Must call fit before get_feature_info")

        output = []
        for prog in self.hall_of_fame:
            formula = render_prog(prog.individual)
            if simplify:
                simplified_formula = simplify_prog_str(formula)
            else:
                simplified_formula = formula
            output.append(
                {
                    "name": prog.name,
                    "formula": simplified_formula,
                    "raw_formula": formula,
                    "fitness": prog.fitness,
                }
            )

        return pd.DataFrame(output)

    def plot_history(
        self, ax: Union[matplotlib.axes._axes.Axes, None] = None
    ) -> matplotlib.axes._axes.Axes:
        """
        Plot the history of the parsimony coefficient and fitness score over generations.

        Args
        ----
        ax : matplotlib.axes._axes.Axes, optional
            The axis to plot on.

        return
        ------
        matplotlib.axes._axes.Axes
            The axis with the history plot.
        """
        if not self.fit_called:
            raise ValueError("Must call fit before plot_history")

        df = pd.DataFrame(self.history)

        if ax is None:
            _, ax1 = plt.subplots()
        else:
            ax1 = ax

        ax2 = ax1.twinx()

        ax1.plot(
            df["generation"],
            df["parsimony"],
            color="tab:blue",
            label="Parsimony Coefficient",
        )
        ax1.set_ylabel("Parsimony Coefficient", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        # Plot Fitness Score
        ax2.plot(
            df["generation"],
            df["best_score"],
            color="tab:orange",
            label="Fitness Score",
        )
        ax2.set_ylabel("Fitness Score", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        # X-axis label and title
        ax1.set_xlabel("Generation")
        ax1.set_title("Parsimony and Fitness over Generations")

        # Optional early stopping marker
        if self.early_termination_counter >= self.early_termination_iters:
            stop_gen = df["generation"].iloc[-1]
            ax1.axvline(x=stop_gen, color="grey", linestyle="--", alpha=0.6)
            ax1.text(
                stop_gen,
                ax1.get_ylim()[1],
                "Early Stop",
                color="grey",
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=8,
            )

        # Combine legends
        lines, labels = [], []
        for axis in [ax1, ax2]:
            line, label = axis.get_legend_handles_labels()
            lines.extend(line)
            labels.extend(label)
        ax1.legend(lines, labels, loc="upper right")

        return ax1
