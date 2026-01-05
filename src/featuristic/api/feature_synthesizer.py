"""
Feature Synthesizer using Genetic Programming.

This module provides a sklearn-compatible API for automated feature synthesis
using symbolic regression and genetic programming.
"""

from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.utils.validation import check_array, column_or_1d

    SKLEARN_AVAILABLE = True

    # Create base class combination
    class _FeatureSynthesizerBase(BaseEstimator, TransformerMixin):
        pass

except ImportError:
    SKLEARN_AVAILABLE = False

    # Create dummy functions and base class
    def check_array(array, **kwargs):
        return np.asarray(array, dtype=np.float64)

    def column_or_1d(y, **kwargs):
        return np.asarray(y).flatten()

    # Single object base when sklearn is not available
    class _FeatureSynthesizerBase:
        pass


class FeatureSynthesizer(_FeatureSynthesizerBase):
    """
    Automated feature synthesis using genetic programming.

    Generates new features by evolving symbolic expressions that
    maximize predictive power while minimizing complexity.

    This class follows scikit-learn's API conventions with fit() and transform()
    methods, making it easy to use in sklearn pipelines.

    Parameters
    ----------
    n_features : int, default=10
        Number of synthesized features to output
    population_size : int, default=50
        Size of the population for genetic programming
    generations : int, default=25
        Maximum number of generations to evolve
    fitness : str or callable, default="auto"
        Fitness function to use. Can be:
        - "auto": Automatically detect based on target type
        - "mse": Mean squared error (regression)
        - "r2": R-squared (regression)
        - "log_loss": Logarithmic loss (classification)
        - "accuracy": Accuracy (classification)
        - Or any custom callable with signature (y_true, y_pred) -> float
    parsimony_coefficient : float, default=0.001
        Penalty coefficient for complex expressions (prevents bloat)
    selection_method : str, default="mrmr"
        Method to select final features from the population:
        - "mrmr": Maximum Relevance Minimum Redundancy
        - "best": Select the k best programs by fitness score
    tournament_size : int, default=5
        Tournament size for genetic selection
    crossover_prob : float, default=0.75
        Probability of crossover during evolution
    mutation_prob : float, default=0.25
        Probability of mutation during evolution
    early_stopping : bool, default=True
        Whether to use early stopping if no improvement
    early_stopping_patience : int, default=5
        Number of generations to wait for improvement before stopping
    random_state : int, optional
        Random seed for reproducibility
    verbose : bool, default=False
        Whether to print progress information

    Attributes
    ----------
    population_ : Population
        The evolved population (available after fitting)
    feature_names_in_ : List[str]
        Names of the input features (available after fitting)
    best_programs_ : List[dict]
        The best programs selected by the selection method
    history_ : List[dict]
        Evolution history tracking best fitness per generation
    is_fitted_ : bool
        Whether the synthesizer has been fitted

    Examples
    --------
    >>> import numpy as np
    >>> from featuristic import FeatureSynthesizer
    >>> X = np.random.randn(100, 3)
    >>> y = X[:, 0] + 0.5 * X[:, 1]
    >>> synth = FeatureSynthesizer(n_features=5, generations=10)
    >>> X_new = synth.fit_transform(X, y)
    >>> X_new.shape
    (100, 5)

    >>> # Use in sklearn pipeline
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> pipeline = Pipeline([
    ...     ('synth', FeatureSynthesizer(n_features=10)),
    ...     ('model', RandomForestRegressor()),
    ... ])
    >>> pipeline.fit(X, y)
    """

    def __init__(
        self,
        n_features: int = 10,
        population_size: int = 50,
        generations: int = 25,
        fitness: Union[str, Callable] = "auto",
        parsimony_coefficient: float = 0.001,
        selection_method: str = "mrmr",
        tournament_size: int = 5,
        crossover_prob: float = 0.75,
        mutation_prob: float = 0.25,
        early_stopping: bool = True,
        early_stopping_patience: int = 5,
        random_state: int = None,
        verbose: bool = False,
    ):
        self.n_features = n_features
        self.population_size = population_size
        self.generations = generations
        self.fitness = fitness
        self.parsimony_coefficient = parsimony_coefficient
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.verbose = verbose

        # Internal state (set during fit)
        self.population_ = None
        self.feature_names_in_ = None
        self.best_programs_ = None
        self.history_ = None
        self.is_fitted_ = False

    def _infer_fitness_function(self, y: np.ndarray) -> str:
        """
        Infer appropriate fitness function based on target type.

        Parameters
        ----------
        y : np.ndarray
            Target values

        Returns
        -------
        str
            Name of the fitness function to use
        """
        if not SKLEARN_AVAILABLE:
            return "mse"

        from sklearn.utils.multiclass import type_of_target

        target_type = type_of_target(y)

        if target_type in ["binary", "multiclass"]:
            return "log_loss"
        elif target_type == "continuous":
            return "mse"
        else:
            return "mse"  # Default

    def _validate_data(
        self, X: Union[np.ndarray, pd.DataFrame], reset: bool = True
    ) -> np.ndarray:
        """
        Validate and convert input data to numpy array.

        Parameters
        ----------
        X : array-like
            Input features
        reset : bool
            Whether to reset the feature names

        Returns
        -------
        np.ndarray
            Validated data as float64 numpy array
        """
        if isinstance(X, pd.DataFrame):
            if reset or self.feature_names_in_ is None:
                self.feature_names_in_ = list(X.columns)
            X_array = X.values.astype(np.float64)
        else:
            if reset or self.feature_names_in_ is None:
                n_features = (
                    X.shape[1]
                    if hasattr(X, "shape") and len(X.shape) == 2
                    else len(X[0])
                    if hasattr(X[0], "__len__")
                    else 1
                )
                self.feature_names_in_ = [f"x{i}" for i in range(n_features)]
            X_array = np.asarray(X, dtype=np.float64)

        if SKLEARN_AVAILABLE:
            X_array = check_array(X_array, ensure_2d=True, dtype=np.float64)

        return X_array

    def _validate_target(self, y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Validate and convert target to numpy array.

        Parameters
        ----------
        y : array-like
            Target values

        Returns
        -------
        np.ndarray
            Validated target as float64 numpy array
        """
        if isinstance(y, pd.Series):
            y_array = y.values.astype(np.float64)
        else:
            y_array = np.asarray(y, dtype=np.float64)

        if SKLEARN_AVAILABLE:
            y_array = column_or_1d(y_array, warn=True).astype(np.float64)

        return y_array

    def fit(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> "FeatureSynthesizer":
        """
        Fit the synthesizer by evolving a population of symbolic programs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input features
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self : FeatureSynthesizer
            Fitted synthesizer
        """
        # Import here to avoid circular dependency
        import featuristic
        from featuristic.fitness import resolve_fitness_function

        # Validate inputs
        X_array = self._validate_data(X, reset=True)
        y_array = self._validate_target(y)

        # Auto-detect fitness function if needed
        fitness_func = self.fitness
        if isinstance(fitness_func, str) and fitness_func == "auto":
            fitness_func = self._infer_fitness_function(y_array)
            if self.verbose:
                print(f"Auto-detected fitness function: {fitness_func}")

        # Resolve fitness function
        fitness_fn = resolve_fitness_function(fitness_func)

        # Create population
        self.population_ = featuristic.Population(
            population_size=self.population_size,
            feature_names=self.feature_names_in_,
            _operations=[],  # Not used, uses builtins
            tournament_size=self.tournament_size,
            crossover_prob=self.crossover_prob,
            mutation_prob=self.mutation_prob,
            seed=self.random_state,
        )

        if self.verbose:
            print(f"Evolving population for {self.generations} generations...")

        # Evolve using extended Population methods
        result = self.population_.evolve_generations(
            X_array,
            y_array,
            fitness_func=fitness_fn,
            n_generations=self.generations,
            parsimony_coefficient=self.parsimony_coefficient,
            early_stopping=self.early_stopping,
            early_stopping_patience=self.early_stopping_patience,
            show_progress_bar=self.verbose,
        )

        # Store results
        self.history_ = result

        # Select best features
        if self.selection_method == "mrmr":
            # Get predictions from all programs
            predictions = self.population_.evaluate_parallel(X_array)
            X_augmented = np.column_stack(predictions)

            # Use mRMR to select
            selected_indices = featuristic.mrmr_select(
                X_augmented,
                y_array,
                num_features=self.n_features,
            )

            all_trees = self.population_.get_trees()
            self.best_programs_ = [all_trees[i] for i in selected_indices]

        elif self.selection_method == "best":
            # Just get the k best by fitness
            best_with_meta = self.population_.get_best(k=self.n_features)
            self.best_programs_ = [bp["tree"] for bp in best_with_meta]

        else:
            raise ValueError(
                f"Unknown selection_method: {self.selection_method}. "
                f"Must be 'mrmr' or 'best'"
            )

        self.is_fitted_ = True

        if self.verbose:
            print(f"Selected {len(self.best_programs_)} features")
            print(f"Best fitness: {result['best_fitness']:.6f}")

        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform input data by evaluating synthesized features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features to transform

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Transformed features

        Raises
        ------
        RuntimeError
            If fit() has not been called yet
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "This FeatureSynthesizer is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

        # Validate input
        X_array = self._validate_data(X, reset=False)

        # Evaluate each best program
        import featuristic

        features = []
        for tree in self.best_programs_:
            feature = featuristic.evaluate_tree(tree, X_array)
            features.append(feature)

        return np.column_stack(features)

    def fit_transform(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """
        Fit the synthesizer and transform input data in one call.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input features
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Transformed features
        """
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self) -> List[str]:
        """
        Get names of synthesized features.

        Returns
        -------
        List[str]
            Names of the output features

        Raises
        ------
        RuntimeError
            If fit() has not been called yet
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "This FeatureSynthesizer is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

        return [f"synth_feature_{i}" for i in range(len(self.best_programs_))]

    def get_programs(self) -> List[dict]:
        """
        Get the best programs with metadata.

        Returns
        -------
        List[dict]
            List of program dictionaries with:
            - 'tree': Tree structure (dict)
            - 'expression': String representation (str)
            - 'depth': Tree depth (int)
            - 'node_count': Number of nodes (int)

        Raises
        ------
        RuntimeError
            If fit() has not been called yet
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "This FeatureSynthesizer is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

        import featuristic

        # Get format strings for readable expressions
        format_strings = self._get_builtin_format_strings()

        programs = []
        for tree in self.best_programs_:
            # Try to use format strings for better expression
            try:
                expression = featuristic.tree_to_string_with_format(
                    tree, format_strings
                )
            except (AttributeError, TypeError):
                # Fall back to basic string representation
                expression = featuristic.tree_to_string(tree)

            programs.append(
                {
                    "tree": tree,
                    "expression": expression,
                    "depth": featuristic.tree_depth(tree),
                    "node_count": featuristic.tree_node_count(tree),
                }
            )

        return programs

    def _get_builtin_format_strings(self) -> List[str]:
        """
        Get format strings for builtin operations.

        Returns
        -------
        List[str]
            Format strings indexed by op_id (use {0}, {1}, etc. for positional args)
        """
        # Format strings for default builtins (must match rust/featuristic-core/src/builtins.rs)
        return [
            "({0} + {1})",  # 0: add
            "({0} - {1})",  # 1: subtract
            "({0} * {1})",  # 2: multiply
            "({0} / {1})",  # 3: divide
            "min({0}, {1})",  # 4: min
            "max({0}, {1})",  # 5: max
            "sin({0})",  # 6: sin
            "cos({0})",  # 7: cos
            "tan({0})",  # 8: tan
            "exp({0})",  # 9: exp
            "log({0})",  # 10: log
            "sqrt({0})",  # 11: sqrt
            "abs({0})",  # 12: abs
            "(-{0})",  # 13: neg
            "({0}^2)",  # 14: square
            "({0}^3)",  # 15: cube
            "clip({0}, {1}, {2})",  # 16: clip
        ]

    def plot_convergence(
        self,
        figsize: Tuple[int, int] = (10, 6),
        title: Optional[str] = None,
        ax: Optional[Any] = None,
    ) -> Any:
        """
        Plot fitness convergence over generations.

        Parameters
        ----------
        figsize : tuple of int, default=(10, 6)
            Figure size (width, height) in inches
        title : str, optional
            Custom title for the plot. If None, uses default title
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, creates new figure

        Returns
        -------
        matplotlib.figure.Figure or matplotlib.axes.Axes
            The figure object if ax is None, otherwise the axes object

        Raises
        ------
        RuntimeError
            If fit() has not been called yet
        ImportError
            If matplotlib is not installed

        Examples
        --------
        >>> synth = FeatureSynthesizer(generations=50)
        >>> synth.fit(X, y)
        >>> fig = synth.plot_convergence()
        >>> fig.savefig('convergence.png')
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "This FeatureSynthesizer is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install it with: pip install matplotlib"
            )

        # Extract fitness history
        fitness_history = self.history_.get("fitness_history", [])
        if not fitness_history:
            raise ValueError(
                "No fitness history available. "
                "The model may have been fitted with an older version."
            )

        # Create figure if ax not provided
        if ax is None:
            was_interactive = plt.isinteractive()
            if was_interactive:
                plt.ioff()  # Turn off interactive mode to prevent double display in Jupyter
            fig, ax = plt.subplots(figsize=figsize)
            if was_interactive:
                plt.ion()  # Restore interactive mode
            return_fig = True
        else:
            return_fig = False

        # Plot fitness over generations
        generations = range(len(fitness_history))
        ax.plot(generations, fitness_history, linewidth=2, marker="o", markersize=4)

        # Add labels and title
        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Best Fitness", fontsize=12)

        if title is None:
            title = "Fitness Convergence Over Generations"
        ax.set_title(title, fontsize=14, fontweight="bold")

        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle="--")

        # Mark best generation
        best_gen = self.history_.get("best_generation", 0)
        best_fitness = self.history_.get("best_fitness", fitness_history[best_gen])
        ax.axvline(
            x=best_gen,
            color="r",
            linestyle="--",
            alpha=0.5,
            label=f"Best (gen {best_gen})",
        )
        ax.scatter([best_gen], [best_fitness], color="r", s=100, zorder=5)

        # Add legend
        ax.legend()

        # Tight layout
        if return_fig:
            fig.tight_layout()
            return fig
        else:
            return ax

    # sklearn compatibility methods
    def _more_tags(self):
        """Tags for sklearn compatibility."""
        return {
            "requires_y": True,
            "requires_positive_X": False,
            "X_types": ["2darray"],
        }

    def __sklearn_is_fitted__(self):
        """Sklearn check if fitted."""
        return self.is_fitted_

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for sklearn compatibility."""
        return {
            "n_features": self.n_features,
            "population_size": self.population_size,
            "generations": self.generations,
            "fitness": self.fitness,
            "parsimony_coefficient": self.parsimony_coefficient,
            "selection_method": self.selection_method,
            "tournament_size": self.tournament_size,
            "crossover_prob": self.crossover_prob,
            "mutation_prob": self.mutation_prob,
            "early_stopping": self.early_stopping,
            "early_stopping_patience": self.early_stopping_patience,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }

    def set_params(self, **params) -> "FeatureSynthesizer":
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self
