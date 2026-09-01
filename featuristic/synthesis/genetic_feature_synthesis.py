"""Contains the SymbolicFeatureGenerator class."""

import os
import random
from collections.abc import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm

from ..constants import SYNTHESIS_FITNESS_METRICS, synthesis_op_kinds
from ..featuristic_lib import runMultipleGAsArray
from ..validation import nonnegative_int, positive_int, probability
from .engine import (
    deserialize_program,
    evaluate_programs,
    run_multiple_gas_python_fitness,
)
from .mrmr import MaxRelevanceMinRedundancy
from .preprocess import preprocess_data
from .render import render_prog, simplify_program
from .symbolic_functions import AVAILABLE_OPERATIONS
from .utils import as_fortran_xy


class GeneticFeatureSynthesis(BaseEstimator, TransformerMixin):
    """
    The Genetic Feature Synthesis class uses genetic programming to generate new
    features using a technique based on Symbolic Regression. This is done by initially
    building a population of naive random formulas that represent transformations of
    the input features. The population is then evolved over a number of generations
    using genetic functions such as mutation and crossover to find the best programs
    that minimize a given fitness function. The best features are then identified using
    a Maximum Relevance Minimum Redundancy (mRMR) algorithm to find those features
    that are most correlated with the target variable while being least correlated with
    each other.

    Notes
    -----
    **Performance Architecture:**
    - This class uses a hybrid Python-Nim architecture for maximum performance
    - The Nim backend reduces Python overhead through:
      * Pre-allocated buffer pools (no per-node allocations)
      * Zero-copy NumPy array access
      * Stack-based evaluation (no Python recursion overhead)
    - The entire genetic algorithm loop runs in Nim when using the default Pearson
      fitness. Pass ``fitness_function`` to score each program in Python; Nim still
      evaluates formulas and evolves the population each generation.

    **Thread Safety:**
    - Nim backend is already faster than Python-based parallelism
    - True multiprocessing is not supported because:
      * Compiled extension state and native arrays are not picklable across processes
      * Reconstructing data structures in worker processes would negate performance gains
    - For better performance, consider running multiple instances with different random seeds

    **Categorical Data Handling:**
    - Non-numeric columns (object, string, or categorical dtypes) are automatically detected
      and encoded during `fit()`.
    - A hybrid encoding strategy is used to preserve the dimensionality of the genetic
      search space:
      * Binary categories (k=2 unique values) are encoded as 0.0 and 1.0 using
        `OrdinalEncoder`. This preserves the binary nature without expanding dimensions.
      * High cardinality categories (k>2 unique values) are encoded into continuous values
        using `TargetEncoder`. Each category is replaced with the mean of the target
        variable for that category.
    - This approach avoids the dimensionality explosion caused by One-Hot Encoding,
      which would create many new columns and complicate the genetic search space.
    - During `transform()`, the same encoders are applied to new data, ensuring
      consistent encoding between training and inference.
    - The target encoder automatically detects if the target is continuous (regression)
      or binary (classification) and adjusts accordingly.
    """

    def __init__(
        self,
        n_features: int = 10,
        population_size: int = 100,
        max_generations: int = 25,
        tournament_size: int = 10,
        crossover_proba: float = 0.85,
        parsimony_coefficient: float = 0.001,
        early_termination_iters: int = 15,
        functions: list[str] | None = None,
        return_all_features: bool = True,
        verbose: bool = False,
        random_state: int | None = None,
        max_depth: int = 6,
        fitness_function: Callable | None = None,
        fitness_metric: str = "pearson",
        pbar: bool = True,
    ):
        """
        Initialize the Symbolic Feature Generator.

        Args
        ----
        n_features : int
            Target number of synthetic features. Internally, `3 * n_features`
            candidates are generated and up to `n_features` non-trivial transformations
            are selected with Maximum Relevance Minimum Redundancy (mRMR).

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
            Complexity penalty for built-in Nim fitness (ignored with
            ``fitness_function``). Pearson uses ``score / size**c``. MAE/MSE use
            ``score * (1 + c * size)`` after linear scaling of ``y_pred``.

        early_termination_iters : int
            If the best score does not improve for this number of generations, then the
            algorithm will terminate early.

        functions : list
            Operator names allowed in synthesized formulas (see
            ``list_symbolic_functions``). These are passed into the Nim GA.
            If ``None``, every built-in operator is used. The leaf ``feature``
            name is ignored if present.

        return_all_features : bool
            If True, transformed data contains the original columns plus selected
            synthetic features. If False, it contains only synthetic features.

        verbose : bool
            Whether to print out aditional information

        random_state : int, optional
            Seed for random number generator for reproducibility. If None,
            results will not be reproducible. Default is None.

        max_depth : int
            The maximum depth of the expression trees in the genetic programming.
            Larger values allow for more complex features but increase the risk of
            overfitting (bloat). Typical values are 3-6. Default is 6.

        fitness_function : callable, optional
            Custom loss for synthesis. Called as
            ``fitness_function(y_true, y_pred, n_nodes) -> float`` once per
            program each generation (lower is better). ``y_true`` / ``y_pred``
            are 1-D float64 arrays; ``n_nodes`` is program size (for parsimony).
            When omitted, fitness is Pearson correlation entirely in Nim (faster).
            With a custom function, GAs run sequentially: Nim still evaluates
            programs and evolves the population.

        fitness_metric : str
            Built-in Nim fitness when ``fitness_function`` is None: ``"pearson"``
            (default), ``"mae"``, or ``"mse"``. MAE/MSE linearly scale ``y_pred``
            before scoring. Ignored when ``fitness_function`` is set.

        pbar : bool
            Whether to display progress bars while generating and selecting
            synthetic features. Default is True.

        """
        positive_int("n_features", n_features)
        positive_int("population_size", population_size, minimum=2)
        positive_int("max_generations", max_generations)
        positive_int("tournament_size", tournament_size)
        probability("crossover_proba", crossover_proba)
        nonnegative_int("early_termination_iters", early_termination_iters)
        positive_int("max_depth", max_depth)
        if parsimony_coefficient < 0:
            raise ValueError("parsimony_coefficient must be non-negative")
        if fitness_function is not None and not callable(fitness_function):
            raise TypeError("fitness_function must be callable or None")

        if functions is None:
            resolved_functions = list(AVAILABLE_OPERATIONS)
        else:
            if not functions:
                raise ValueError("functions must contain at least one operator")
            # Validate function names
            for func in functions:
                if func not in AVAILABLE_OPERATIONS:
                    raise ValueError(
                        f"Function '{func}' not found in symbolic operations"
                    )
            if not any(func != "feature" for func in functions):
                raise ValueError(
                    "functions must contain at least one non-leaf operator"
                )
            resolved_functions = functions

        metric_key = fitness_metric.lower()
        if metric_key not in SYNTHESIS_FITNESS_METRICS:
            raise ValueError(
                f"fitness_metric must be one of {sorted(SYNTHESIS_FITNESS_METRICS)}, "
                f"got {fitness_metric!r}"
            )
        # Estimator parameters must be stored unchanged for sklearn.clone().
        self.functions = functions
        self.fitness_metric = fitness_metric
        self.op_kinds_ = synthesis_op_kinds(resolved_functions)

        self.population_size = population_size
        self.max_generations = max_generations
        self.tournament_size = tournament_size
        self.crossover_proba = crossover_proba
        self.n_features = n_features
        self.parsimony_coefficient = parsimony_coefficient
        self.max_depth = max_depth

        self.history = []
        self.hall_of_fame = []

        self.early_termination_iters = early_termination_iters

        self.return_all_features = return_all_features

        self.verbose = verbose
        self.random_state = random_state
        self.fitness_function = fitness_function
        self.pbar = pbar

        # Categorical encoding attributes
        self.target_encoder_ = None
        self.binary_encoder_ = None
        self.high_card_cols_ = []
        self.binary_cols_ = []
        self.synthetic_feature_stats_ = {}

    def _select_best_features(self, X: pd.DataFrame, y: pd.Series):
        """
        Select the best features using the mRMR algorithm.

        Combines original features with synthetic features from hall of fame,
        then runs mRMR on ALL features to select the best combination.

        Args
        ----
        X : pd.DataFrame
            The dataframe with the original features.

        y : pd.Series
            The target variable.

        return
        ------
        None
        """
        # Evaluate synthetic features from hall of fame
        programs = [entry["individual"] for entry in self.hall_of_fame]
        synthetic_features = evaluate_programs(X, programs)

        # Clean NaN/Inf values from synthetic features
        synthetic_features = self._clean_features(synthetic_features)

        # Name synthetic features
        actual_hof_size = len(self.hall_of_fame)
        synthetic_features.columns = [f"synth_{i}" for i in range(actual_hof_size)]

        # Ensure X is a DataFrame before concatenation
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        # Clean the combined features to handle any remaining NaN/Inf values
        synthetic_features = self._clean_features(
            synthetic_features, fit_synthetic_scaler=True
        )

        # Run mRMR on synthetic features only to select the best ones
        # (We want n_features synthetic features, not total features)
        if len(synthetic_features.columns) >= self.n_features:
            selected_names = (
                MaxRelevanceMinRedundancy(k=self.n_features, pbar=self.pbar)
                .fit_transform(synthetic_features, y)
                .columns
            )
        else:
            # If we generated fewer than n_features (due to filtering), return all
            selected_names = synthetic_features.columns

        # Map selected synthetic feature names back to original indices
        selected_synth_names = [
            name for name in selected_names if str(name).startswith("synth_")
        ]

        # Match the public return_all_features contract: either retain the original
        # columns alongside the selected synthetic features, or return only the
        # generated features.
        if self.return_all_features:
            selected_names = list(X_df.columns) + selected_synth_names
        else:
            selected_names = selected_synth_names

        # Filter hall of fame to only include selected synthetic features
        selected_hof = []
        for name in selected_names:
            # Convert to string if needed (mRMR might return integer column names)
            name_str = str(name) if not isinstance(name, str) else name

            if name_str.startswith("synth_"):
                # Extract index from name (e.g., "synth_5" -> 5)
                idx = int(name_str.split("_")[1])
                if idx < len(self.hall_of_fame):
                    selected_hof.append(self.hall_of_fame[idx])
                    # Update the name to match what was selected
                    selected_hof[-1]["name"] = name_str

        # Store the selected feature names (including original features)
        self.selected_feature_names_ = list(selected_names)

        # Store full hall of fame before filtering (for inspection)
        self.all_generated_features_ = self.hall_of_fame.copy()

        # Update hall of fame to only contain selected synthetic features
        self.hall_of_fame = selected_hof

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GeneticFeatureSynthesis":
        """
        Fit the symbolic feature generator to the data.

        Uses Nim backend for all genetic algorithm operations with SINGLE CALL OPTIMIZATION.

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
        # Convert numpy arrays to pandas if needed, then reset index
        X_array = np.asarray(X)
        y_array = np.asarray(y)
        if X_array.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if y_array.ndim != 1:
            raise ValueError("y must be 1-dimensional")
        if X_array.shape[0] == 0 or X_array.shape[1] == 0:
            raise ValueError("X must contain at least one row and one feature")
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if isinstance(X, pd.DataFrame) and X.columns.duplicated().any():
            raise ValueError("X must not contain duplicate column names")

        # set_params() may change search parameters after construction.
        resolved_functions = (
            list(AVAILABLE_OPERATIONS) if self.functions is None else self.functions
        )
        if not resolved_functions or not any(
            func != "feature" for func in resolved_functions
        ):
            raise ValueError("functions must contain at least one non-leaf operator")
        unknown_functions = set(resolved_functions) - set(AVAILABLE_OPERATIONS)
        if unknown_functions:
            raise ValueError(f"Unknown symbolic functions: {sorted(unknown_functions)}")
        metric_key = self.fitness_metric.lower()
        if metric_key not in SYNTHESIS_FITNESS_METRICS:
            raise ValueError(
                f"fitness_metric must be one of {sorted(SYNTHESIS_FITNESS_METRICS)}, "
                f"got {self.fitness_metric!r}"
            )
        self.op_kinds_ = synthesis_op_kinds(resolved_functions)

        X_pd = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y_pd = pd.Series(y) if not isinstance(y, pd.Series) else y

        X_copy, y_copy = preprocess_data(
            X_pd.reset_index(drop=True), y_pd.reset_index(drop=True)
        )

        # A refit must learn normalization parameters from the new training data.
        self.synthetic_feature_stats_ = {}

        # Validate input data for NaN/Inf values (raises error if found)
        self._clean_features(X_copy, is_input=True)

        # Handle categorical columns
        # Identify non-numeric columns
        non_numeric_cols = [
            col for col in X_copy.columns if not is_numeric_dtype(X_copy[col])
        ]

        if non_numeric_cols:
            # Split into binary (k=2) and high cardinality (k>2) columns
            self.binary_cols_ = [
                col for col in non_numeric_cols if X_copy[col].nunique() == 2
            ]
            self.high_card_cols_ = [
                col for col in non_numeric_cols if X_copy[col].nunique() > 2
            ]

            # Encode binary columns with OrdinalEncoder (maps to 0.0 and 1.0)
            if self.binary_cols_:
                self.binary_encoder_ = OrdinalEncoder(dtype=np.float64)
                X_copy[self.binary_cols_] = self.binary_encoder_.fit_transform(
                    X_copy[self.binary_cols_]
                )

            # Encode high cardinality columns with TargetEncoder
            if self.high_card_cols_:
                target_type = "continuous" if is_numeric_dtype(y_copy) else "binary"
                self.target_encoder_ = TargetEncoder(target_type=target_type)
                X_copy[self.high_card_cols_] = self.target_encoder_.fit_transform(
                    X_copy[self.high_card_cols_], y_copy
                )

        # Store feature names for later deserialization
        self.feature_names_ = X_copy.columns.tolist()

        # Set random seeds for reproducibility
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        # Generate diverse features using Nim GA
        generations_per_ga = self.max_generations

        X_f, y_c = as_fortran_xy(X_copy, y_copy)

        # Generate a larger candidate pool before mRMR selection. Individual GAs can
        # converge to a raw input feature; those candidates are intentionally discarded.
        # Over-generation preserves the contract that n_features is the requested
        # number of actual synthetic transformations, rather than merely an upper bound.
        num_candidate_features = self.n_features * 3

        # Generate random seeds for each GA
        if self.random_state is not None:
            # Generate deterministic seeds for reproducibility
            random_seeds = [
                (self.random_state + i) % (2**31) for i in range(num_candidate_features)
            ]
        else:
            # Generate random seeds
            random_seeds = [
                random.randint(0, 2**31 - 1) for _ in range(num_candidate_features)
            ]

        # Note: Nim function returns tuple (positional args due to nimpy)
        if self.fitness_function is None:
            batch_size = (
                min(num_candidate_features, os.cpu_count() or 1)
                if self.pbar
                else num_candidate_features
            )
            native_results = [[] for _ in range(8)]
            with tqdm(
                total=num_candidate_features,
                desc="Generating synthetic features...",
                disable=not self.pbar,
            ) as progress:
                for batch_start in range(0, num_candidate_features, batch_size):
                    batch_seeds = random_seeds[batch_start : batch_start + batch_size]
                    batch_results = runMultipleGAsArray(
                        X_f,
                        y_c,
                        len(batch_seeds),
                        generations_per_ga,
                        self.population_size,
                        self.max_depth,
                        self.tournament_size,
                        self.crossover_proba,
                        self.parsimony_coefficient,
                        batch_seeds,
                        self.op_kinds_,
                        SYNTHESIS_FITNESS_METRICS[self.fitness_metric.lower()],
                        self.early_termination_iters,
                    )
                    for combined, batch_values in zip(
                        native_results, batch_results, strict=True
                    ):
                        combined.extend(batch_values)
                    progress.update(len(batch_seeds))

            (
                best_feature_indices,
                best_op_kinds,
                best_left_children,
                best_right_children,
                best_constants,
                best_fitnesses,
                best_scores,
                generation_histories,
            ) = native_results
        else:
            with tqdm(
                total=num_candidate_features,
                desc="Generating synthetic features...",
                disable=not self.pbar,
            ) as progress:
                custom = run_multiple_gas_python_fitness(
                    X_f,
                    y_c,
                    num_candidate_features,
                    generations_per_ga,
                    self.population_size,
                    self.max_depth,
                    self.tournament_size,
                    self.crossover_proba,
                    random_seeds,
                    self.fitness_function,
                    self.op_kinds_,
                    self.early_termination_iters,
                    progress_callback=progress.update,
                )
            best_feature_indices = custom["best_feature_indices"]
            best_op_kinds = custom["best_op_kinds"]
            best_left_children = custom["best_left_children"]
            best_right_children = custom["best_right_children"]
            best_constants = custom["best_constants"]
            best_fitnesses = custom["best_fitnesses"]
            best_scores = custom["best_scores"]
            generation_histories = custom["generation_histories"]

        # Store generation histories for convergence plotting
        self.generation_histories_ = generation_histories

        # Process results from Nim
        self.hall_of_fame = []

        for feature_idx in range(num_candidate_features):
            # Extract serialized program data for this GA
            program_data = {
                "feature_indices": best_feature_indices[feature_idx],
                "op_kinds": best_op_kinds[feature_idx],
                "left_children": best_left_children[feature_idx],
                "right_children": best_right_children[feature_idx],
                "constants": best_constants[feature_idx],
                "fitness": best_fitnesses[feature_idx],
                "score": best_scores[feature_idx],
            }

            best_fitness = program_data["fitness"]

            # Deserialize for formula string generation (only for display)
            best_program_for_display = deserialize_program(
                program_data, self.feature_names_
            )
            formula = render_prog(best_program_for_display)

            # Filter out programs that simplify to single features (no actual transformation)
            # Check if the simplified program has any operations (has children)

            simplified_program = simplify_program(best_program_for_display)

            # If the program simplifies to just a feature (no children), skip it
            if "children" not in simplified_program:
                # This is just a raw feature, not a synthetic transformation
                # Generate a warning and continue to next feature
                if self.verbose:
                    print(
                        f"Warning: Feature {feature_idx} simplified to raw feature, skipping..."
                    )
                continue

            # Add to hall of fame
            self.hall_of_fame.append(
                {
                    "individual": program_data,
                    "fitness": best_fitness,
                    "formula": formula,
                    "name": f"synth_{feature_idx}",
                }
            )

            # Track history
            self.history.append(
                {
                    "feature": feature_idx,
                    "best_fitness": best_fitness,
                    "best_program": formula,
                }
            )

            if self.verbose and feature_idx == 0:
                print(f"First generated feature: {formula}")
                print(f"Fitness: {best_fitness:.6f}")

        if self.verbose and len(self.hall_of_fame) < self.n_features:
            print(
                f"Generated {len(self.hall_of_fame)} of {self.n_features} requested "
                "non-trivial synthetic features."
            )

        if self.verbose:
            print(f"Generated {len(self.hall_of_fame)} synthetic features using Nim GA")

        # Select the best features using mRMR
        self._select_best_features(X_copy, y_copy)

        return self

    def _clean_features(
        self,
        df: pd.DataFrame,
        is_input: bool = False,
        fit_synthetic_scaler: bool = False,
    ) -> pd.DataFrame:
        """
        Clean synthetic features by replacing NaN and Inf values.

        For input data (is_input=True), raises ValueError if NaN/Inf values are found.
        For synthetic features (is_input=False), replaces NaN/Inf and normalizes to prevent
        numerical issues with models like LogisticRegression.

        Args
        ----
        df : pd.DataFrame
            The dataframe to clean.

        is_input : bool
            If True, this is input data and should be validated (not auto-imputed).
            If False, this is synthetic features and should be cleaned.

        fit_synthetic_scaler : bool
            If True, learn normalization statistics for synthetic features from this
            training data. Otherwise, reuse the statistics learned during ``fit``.

        return
        ------
        pd.DataFrame
            Cleaned dataframe with NaN/Inf replaced and synthetic features normalized.

        Raises
        ------
        ValueError
            If is_input=True and the dataframe contains NaN or Inf values.
        """
        # Make a copy to avoid modifying the original
        df_clean = df.copy()

        # Check for NaN and Inf values
        has_nan = df_clean.isna().any().any()
        has_inf = np.isinf(df_clean.select_dtypes(include=[np.number])).any().any()

        if is_input:
            # For input data, raise an error instead of silently imputing
            if has_nan or has_inf:
                nan_cols = df_clean.columns[df_clean.isna().any()].tolist()
                inf_cols = (
                    df_clean.select_dtypes(include=[np.number])
                    .columns[
                        np.isinf(df_clean.select_dtypes(include=[np.number])).any()
                    ]
                    .tolist()
                )

                issues = []
                if nan_cols:
                    issues.append(f"NaN values in columns: {nan_cols}")
                if inf_cols:
                    issues.append(f"Inf values in columns: {inf_cols}")

                raise ValueError(
                    "Input data contains NaN or Inf values. "
                    "Please handle missing values before using GeneticFeatureSynthesis. "
                    "Consider using sklearn.impute.SimpleImputer or pandas fillna methods. "
                    f"Issues found: {'; '.join(issues)}"
                )
        else:
            # For synthetic features, clean NaN/Inf as they can legitimately occur
            # Replace Inf and -Inf with NaN first
            df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Fill NaN with 0 for synthetic features only
            df_clean.fillna(0, inplace=True)

        # Clip extreme values to prevent overflow (only numeric columns)
        max_value = 1e6
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].clip(
            lower=-max_value, upper=max_value
        )

        # Normalize synthetic features using statistics learned during fit. Recomputing
        # these values here would make a row's output depend on the rest of its batch.
        for col in df_clean.columns:
            if str(col).startswith("synth_"):
                col_data = df_clean[col].values
                if fit_synthetic_scaler:
                    self.synthetic_feature_stats_[col] = (
                        float(np.mean(col_data)),
                        float(np.std(col_data)),
                    )

                stats = self.synthetic_feature_stats_.get(col)
                if stats is not None:
                    mean, std = stats
                    # Leave constant training features unchanged, as before.
                    if std <= 1e-10:
                        continue
                    df_clean[col] = (col_data - mean) / (std + 1e-10)
                    df_clean[col] = df_clean[col].clip(-10, 10)

        return df_clean

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Transform the dataframe of features using the selected features.

        Returns the features selected by mRMR, which may include both
        original features and synthetic features.

        Args
        ----
        X : pd.DataFrame
            The dataframe with the features.

        return
        ------
        pd.DataFrame
            The transformed dataframe with selected features.
        """
        check_is_fitted(self, "feature_names_")

        # Convert numpy array to pandas DataFrame if needed
        X_pd = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        # Transform categorical columns if encoders were fitted
        if self.binary_cols_ and self.binary_encoder_ is not None:
            X_pd[self.binary_cols_] = self.binary_encoder_.transform(
                X_pd[self.binary_cols_]
            )

        if self.high_card_cols_ and self.target_encoder_ is not None:
            X_pd[self.high_card_cols_] = self.target_encoder_.transform(
                X_pd[self.high_card_cols_]
            )

        # Evaluate synthetic features from hall of fame
        if len(self.hall_of_fame) > 0:
            # Extract programs and evaluate them
            programs = [x["individual"] for x in self.hall_of_fame]
            synthetic_features = evaluate_programs(
                X_pd.reset_index(drop=True), programs
            )

            # Clean NaN/Inf values from synthetic features
            synthetic_features = self._clean_features(synthetic_features)

            synthetic_features.columns = [x["name"] for x in self.hall_of_fame]

            # Combine original and synthetic features
            all_features = pd.concat(
                [X_pd.reset_index(drop=True), synthetic_features], axis=1
            )

            # Clean the combined features to handle any remaining NaN/Inf values
            all_features = self._clean_features(all_features)

            return all_features[self.selected_feature_names_]

        return X_pd.reset_index(drop=True)[self.selected_feature_names_]

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

    def get_feature_info(self) -> pd.DataFrame:
        """
        Get the information about the best programs found.

        return
        ------
        pd.DataFrame
            The dataframe with the information.
        """
        check_is_fitted(self, "feature_names_")

        output = []
        for prog in self.hall_of_fame:
            # Deserialize the program for rendering
            individual = deserialize_program(prog["individual"], self.feature_names_)

            tmp = {
                "name": prog["name"],
                "formula": render_prog(individual),
                "fitness": prog["fitness"],
            }
            output.append(tmp)

        return pd.DataFrame(output)

    def get_programs(self):
        """
        Get raw program structures from hall of fame.

        Returns
        -------
        List[dict]
            Program dictionaries sorted by fitness, each with:
            - 'program': raw program structure
            - 'fitness': fitness score
            - 'formula': string representation
            - 'name': feature name
        """
        check_is_fitted(self, "feature_names_")

        sorted_hof = sorted(self.hall_of_fame, key=lambda x: x["fitness"])
        return [
            {
                "program": entry["individual"],
                "fitness": entry["fitness"],
                "formula": entry["formula"],
                "name": entry["name"],
            }
            for entry in sorted_hof
        ]

    def plot_history(self, ax: matplotlib.axes._axes.Axes | None = None):
        """
        Plot the history of the fitness function with enhanced visualization.

        Displays the convergence of the genetic algorithm across generations,
        showing each GA's convergence and aggregated statistics.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, creates a new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot.

        Examples
        --------
        >>> synth = GeneticFeatureSynthesis(n_features=10)
        >>> synth.fit(X, y)
        >>> ax = synth.plot_history()
        """
        check_is_fitted(self, "feature_names_")

        if ax is None:
            _fig, ax = plt.subplots(figsize=(10, 6))

        # Check if we have generation histories
        if (
            not hasattr(self, "generation_histories_")
            or len(self.generation_histories_) == 0
        ):
            ax.text(
                0.5,
                0.5,
                "No generation history available.\nMake sure the model was fitted after generation tracking was added.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )
            return ax

        raw_histories = [
            np.asarray(history, dtype=float)
            for history in self.generation_histories_
            if len(history) > 0
        ]
        if not raw_histories:
            ax.text(
                0.5,
                0.5,
                "No generation history available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return ax

        # GAs may terminate at different generations. Pad each completed run
        # with its final best score so aggregate curves retain stopped runs.
        num_generations = max(len(history) for history in raw_histories)
        histories = np.empty((len(raw_histories), num_generations), dtype=float)
        for ga_idx, history in enumerate(raw_histories):
            histories[ga_idx, : len(history)] = history
            histories[ga_idx, len(history) :] = history[-1]

        generations = np.arange(num_generations)

        # Calculate statistics across all GAs at each generation
        best_per_gen = np.min(histories, axis=0)  # Best fitness across all GAs
        median_per_gen = np.median(histories, axis=0)  # Median across all GAs
        min_per_gen = np.min(histories, axis=0)  # Same as best
        max_per_gen = np.max(histories, axis=0)  # Worst GA at each generation

        # Plot individual GA convergence curves (faint)
        for history in raw_histories:
            ax.plot(
                np.arange(len(history)),
                history,
                "-",
                linewidth=0.5,
                color="#64748b",
                alpha=0.3,
                zorder=1,
            )

        # Plot shaded region (min-max spread)
        ax.fill_between(
            generations,
            min_per_gen,
            max_per_gen,
            alpha=0.15,
            color="#6366f1",
            label="Min-Max Spread",
            zorder=2,
        )

        # Plot best fitness across all GAs (thick red line)
        ax.plot(
            generations,
            best_per_gen,
            "-",
            linewidth=2.5,
            color="#dc2626",
            alpha=0.9,
            label="Best Fitness",
            zorder=4,
        )

        # Plot median fitness (thick blue line)
        ax.plot(
            generations,
            median_per_gen,
            "--",
            linewidth=2,
            color="#2563eb",
            alpha=0.8,
            label="Median Fitness",
            zorder=3,
        )

        # Highlight the final best fitness
        final_best = best_per_gen[-1]
        final_gen = generations[-1]
        ax.scatter(
            [final_gen],
            [final_best],
            s=250,
            color="#dc2626",
            marker="*",
            zorder=5,
            edgecolors="white",
            linewidths=2,
            label=f"Best: {final_best:.4f}",
        )

        # Styling
        ax.set_xlabel("Generation", fontsize=12, fontweight="bold")
        ax.set_ylabel("Fitness Score", fontsize=12, fontweight="bold")
        ax.set_title(
            "Feature Synthesis Convergence (per Generation)",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

        # Add grid with better styling
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)

        # Better legend
        ax.legend(
            loc="best",
            framealpha=0.95,
            shadow=True,
            fontsize=10,
            edgecolor="#ddd",
            ncol=2,
        )

        # Add light background
        ax.set_facecolor("#f9fafb")

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        return ax
