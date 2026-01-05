"""
Feature Selector using Genetic Algorithm.

This module provides a sklearn-compatible API for evolutionary feature selection
using binary genomes and genetic algorithms.
"""

from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.utils.validation import check_array, column_or_1d

    SKLEARN_AVAILABLE = True

    # Create base class combination
    class _FeatureSelectorBase(BaseEstimator, TransformerMixin):
        pass

except ImportError:
    SKLEARN_AVAILABLE = False

    # Create dummy functions and base class
    def check_array(array, **kwargs):
        return np.asarray(array, dtype=np.float64)

    def column_or_1d(y, **kwargs):
        return np.asarray(y).flatten()

    # Single object base when sklearn is not available
    class _FeatureSelectorBase:
        pass


class FeatureSelector(_FeatureSelectorBase):
    """
    Selects features using evolutionary algorithm with binary genomes.

    This class uses a genetic algorithm to evolve binary genomes that represent
    feature subsets, optimizing a user-provided objective function (lower is better).

    Parameters
    ----------
    objective_function : callable
        Function to minimize. Signature: objective(X_subset, y) -> float
        Lower values are better (e.g., MSE, negative accuracy, etc.)
    population_size : int, default=50
        Number of individuals in the population
    max_generations : int, default=100
        Maximum number of generations to evolve
    tournament_size : int, default=10
        Tournament size for selection (larger = more selection pressure)
    crossover_proba : float, default=0.9
        Probability of crossover between 0.0 and 1.0
    mutation_proba : float, default=0.1
        Probability of bit-flip mutation between 0.0 and 1.0
    early_stopping : bool, default=True
        Whether to use early stopping if no improvement
    early_stopping_patience : int, default=15
        Number of generations to wait for improvement before stopping
    random_state : int, optional
        Random seed for reproducibility
    verbose : bool, default=False
        Whether to print progress information

    Attributes
    ----------
    population_ : BinaryPopulation
        The evolved population (available after fitting)
    feature_names_in_ : List[str]
        Names of the input features (available after fitting)
    n_features_in_ : int
        Number of input features (available after fitting)
    best_genome_ : np.ndarray
        Best binary genome found (available after fitting)
    best_score_ : float
        Best fitness score found (available after fitting)
    selected_features_ : List[str]
        Names of selected features (available after fitting)
    history_ : dict
        Evolution history tracking best fitness and generation info
    is_fitted_ : bool
        Whether the selector has been fitted

    Examples
    --------
    >>> import numpy as np
    >>> from featuristic import FeatureSelector
    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.metrics import mean_squared_error
    >>>
    >>> X = np.random.randn(100, 10)
    >>> y = X[:, 0] + 0.5 * X[:, 1]
    >>>
    >>> def objective(X_subset, y):
    ...     model = Ridge().fit(X_subset, y)
    ...     return mean_squared_error(y, model.predict(X_subset))
    >>>
    >>> selector = FeatureSelector(
    ...     objective_function=objective,
    ...     population_size=40,
    ...     max_generations=50,
    ...     random_state=42
    ... )
    >>> X_selected = selector.fit_transform(X, y)
    >>> X_selected.shape
    (100, N)  # where N < 10
    >>>
    >>> # Get selected feature names
    >>> selector.selected_features_

    >>> # Use in sklearn pipeline
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.ensemble import RandomForestRegressor
    >>>
    >>> pipeline = Pipeline([
    ...     ('selector', FeatureSelector(
    ...         objective_function=objective,
    ...         population_size=30,
    ...         max_generations=25
    ...     )),
    ...     ('model', RandomForestRegressor()),
    ... ])
    >>> pipeline.fit(X, y)
    """

    def __init__(
        self,
        objective_function: Callable,
        population_size: int = 50,
        max_generations: int = 100,
        tournament_size: int = 10,
        crossover_proba: float = 0.9,
        mutation_proba: float = 0.1,
        early_stopping: bool = True,
        early_stopping_patience: int = 15,
        random_state: int = None,
        verbose: bool = False,
    ):
        self.objective_function = objective_function
        self.population_size = population_size
        self.max_generations = max_generations
        self.tournament_size = tournament_size
        self.crossover_proba = crossover_proba
        self.mutation_proba = mutation_proba
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.verbose = verbose

        # Internal state (set during fit)
        self.population_ = None
        self.feature_names_in_ = None
        self.n_features_in_ = None
        self.best_genome_ = None
        self.best_score_ = None
        self.selected_features_ = None
        self.history_ = None
        self.is_fitted_ = False

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
                self.n_features_in_ = len(self.feature_names_in_)
            X_array = X.values.astype(np.float64)
        else:
            X_array = np.asarray(X, dtype=np.float64)
            if reset or self.feature_names_in_ is None:
                if hasattr(X_array, "shape") and len(X_array.shape) == 2:
                    self.n_features_in_ = X_array.shape[1]
                else:
                    # Assume 1D array or similar
                    self.n_features_in_ = 1
                self.feature_names_in_ = [f"x{i}" for i in range(self.n_features_in_)]

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
    ) -> "FeatureSelector":
        """
        Fit the selector by evolving a population of binary genomes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input features
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self : FeatureSelector
            Fitted selector
        """
        # Import here to avoid circular dependency
        import featuristic

        # Validate inputs
        X_array = self._validate_data(X, reset=True)
        y_array = self._validate_target(y)

        # Create binary population
        self.population_ = featuristic.BinaryPopulation(
            population_size=self.population_size,
            num_features=self.n_features_in_,
            tournament_size=self.tournament_size,
            crossover_prob=self.crossover_proba,
            mutation_prob=self.mutation_proba,
            seed=self.random_state,
        )

        if self.verbose:
            print(f"Evolving population for {self.max_generations} generations...")

        # Evolve using extended BinaryPopulation methods
        result = self.population_.evolve_generations(
            X_array,
            y_array,
            objective_func=self.objective_function,
            n_generations=self.max_generations,
            early_stopping=self.early_stopping,
            early_stopping_patience=self.early_stopping_patience,
            show_progress_bar=self.verbose,
        )

        # Store results
        self.best_genome_ = np.array(self.population_.get_best_genome(), dtype=bool)
        self.best_score_ = result["best_fitness"]
        self.history_ = result
        self.selected_features_ = [
            name
            for selected, name in zip(self.best_genome_, self.feature_names_in_)
            if selected
        ]
        self.is_fitted_ = True

        if self.verbose:
            print(f"Selected {len(self.selected_features_)} features")
            print(f"Best score: {self.best_score_:.6f}")

        return self

    def transform(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Transform input data by selecting best features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features to transform

        Returns
        -------
        np.ndarray or pd.DataFrame
            Transformed features with only selected columns

        Raises
        ------
        RuntimeError
            If fit() has not been called yet
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "This FeatureSelector is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

        # Validate input
        X_array = self._validate_data(X, reset=False)

        # Return DataFrame if input was DataFrame
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        else:
            return X_array[:, self.best_genome_]

    def fit_transform(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Fit the selector and transform input data in one call.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input features
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        np.ndarray or pd.DataFrame
            Transformed features with only selected columns
        """
        self.fit(X, y)
        return self.transform(X)

    def get_support(self, indices: bool = False) -> Union[np.ndarray, List[int]]:
        """
        Get mask or indices of selected features.

        Parameters
        ----------
        indices : bool, default=False
            If True, return feature indices instead of boolean mask

        Returns
        -------
        np.ndarray or List[int]
            Boolean mask or integer indices of selected features

        Raises
        ------
        RuntimeError
            If fit() has not been called yet
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "This FeatureSelector is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

        mask = self.best_genome_
        if indices:
            return np.where(mask)[0].tolist()
        return mask

    def get_feature_names_out(self, input_features: Any = None) -> List[str]:
        """
        Get names of selected output features.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features (not used, kept for sklearn compatibility)

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
                "This FeatureSelector is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

        return self.selected_features_.copy()

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
        >>> selector = FeatureSelector(objective_function=my_objective)
        >>> selector.fit(X, y)
        >>> fig = selector.plot_convergence()
        >>> fig.savefig('convergence.png')
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "This FeatureSelector is not fitted yet. "
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
    def _more_tags(self) -> dict:
        """Tags for sklearn compatibility."""
        return {
            "requires_y": True,
            "requires_positive_X": False,
            "X_types": ["2darray"],
        }

    def __sklearn_is_fitted__(self) -> bool:
        """Sklearn check if fitted."""
        return self.is_fitted_

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for sklearn compatibility."""
        return {
            "objective_function": self.objective_function,
            "population_size": self.population_size,
            "max_generations": self.max_generations,
            "tournament_size": self.tournament_size,
            "crossover_proba": self.crossover_proba,
            "mutation_proba": self.mutation_proba,
            "early_stopping": self.early_stopping,
            "early_stopping_patience": self.early_stopping_patience,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }

    def set_params(self, **params) -> "FeatureSelector":
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self
