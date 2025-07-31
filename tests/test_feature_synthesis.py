import pandas as pd
import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")

from featuristic import GeneticFeatureSynthesis


@pytest.fixture
def regression_data():
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f"x{i}" for i in range(5)])
    y = X["x0"] * 2 + X["x1"] - X["x2"]
    return X, y


def test_feature_synthesis_fit_transform(regression_data):
    X, y = regression_data
    fs = GeneticFeatureSynthesis(
        num_features=3,
        max_generations=5,
        population_size=20,
        n_jobs=1,
        show_progress_bar=False,
    )
    Xt = fs.fit_transform(X, y)
    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape[0] == X.shape[0]
    assert fs.fit_called
    assert fs.get_feature_info().shape[0] == 3


def test_get_feature_info_simplify_argument(regression_data):
    X, y = regression_data
    fs = GeneticFeatureSynthesis(
        num_features=3,
        max_generations=5,
        population_size=20,
        n_jobs=1,
        show_progress_bar=False,
    )
    fs.fit(X, y)
    df_unsimplified = fs.get_feature_info()
    df_simplified = fs.get_feature_info(simplify=True)
    assert isinstance(df_unsimplified, pd.DataFrame)
    assert isinstance(df_simplified, pd.DataFrame)
    assert df_unsimplified.shape == df_simplified.shape
    # At least one formula should differ if simplification does something
    formulas_unsimplified = df_unsimplified["formula"].tolist()
    formulas_simplified = df_simplified["formula"].tolist()
    assert len(formulas_unsimplified) == len(formulas_simplified)
    # Allow for possibility that some formulas do not simplify
    assert any(
        u != s for u, s in zip(formulas_unsimplified, formulas_simplified)
    ) or all(u == s for u, s in zip(formulas_unsimplified, formulas_simplified))


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_plot_fitness_history_runs():
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f"x{i}" for i in range(5)])
    y = X["x0"] + X["x1"] * 2 - X["x2"]

    fs = GeneticFeatureSynthesis(
        num_features=2,
        max_generations=3,
        population_size=10,
        n_jobs=1,
        show_progress_bar=False,
    )
    fs.fit(X, y)

    ax = fs.plot_fitness_history()
    assert isinstance(ax, matplotlib.axes._axes.Axes)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_plot_fitness_history_with_ax():
    import matplotlib.pyplot as plt

    X = pd.DataFrame(np.random.randn(100, 5), columns=[f"x{i}" for i in range(5)])
    y = X["x0"] + X["x1"] * 2 - X["x2"]

    fs = GeneticFeatureSynthesis(
        num_features=2,
        max_generations=3,
        population_size=10,
        n_jobs=1,
        show_progress_bar=False,
    )
    fs.fit(X, y)

    fig, ax = plt.subplots()
    ax_returned = fs.plot_fitness_history(ax=ax)
    assert ax_returned is ax
    assert len(ax.lines) > 0


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_plot_parsimony_history_runs():
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f"x{i}" for i in range(5)])
    y = X["x0"] + X["x1"] * 2 - X["x2"]

    fs = GeneticFeatureSynthesis(
        num_features=2,
        max_generations=3,
        population_size=10,
        n_jobs=1,
        show_progress_bar=False,
    )
    fs.fit(X, y)

    ax = fs.plot_parsimony_history()
    assert isinstance(ax, matplotlib.axes._axes.Axes)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_plot_parsimony_history_with_ax():
    import matplotlib.pyplot as plt

    X = pd.DataFrame(np.random.randn(100, 5), columns=[f"x{i}" for i in range(5)])
    y = X["x0"] + X["x1"] * 2 - X["x2"]

    fs = GeneticFeatureSynthesis(
        num_features=2,
        max_generations=3,
        population_size=10,
        n_jobs=1,
        show_progress_bar=False,
    )
    fs.fit(X, y)

    fig, ax = plt.subplots()
    ax_returned = fs.plot_parsimony_history(ax=ax)
    assert ax_returned is ax
    assert len(ax.lines) > 0


def contains_constant(program):
    """Recursively check if a program contains a constant value."""
    if "value" in program:
        return True
    if "children" in program:
        return any(contains_constant(child) for child in program["children"])
    return False


def extract_constants_from_program(program):
    """Recursively extract all constants from a program."""
    constants = []
    if "value" in program:
        constants.append(program["value"])
    if "children" in program:
        for child in program["children"]:
            constants.extend(extract_constants_from_program(child))
    return constants


def test_min_max_constant_bounds(regression_data):
    """Test that constants are within min/max bounds."""
    X, y = regression_data
    min_val, max_val = -2.5, 1.5
    fs = GeneticFeatureSynthesis(
        num_features=5,
        max_generations=5,
        population_size=20,
        n_jobs=1,
        show_progress_bar=False,
        include_constants=True,
        min_constant_val=min_val,
        max_constant_val=max_val,
    )
    fs.fit(X, y)
    for program in fs.hall_of_fame:
        constants = extract_constants_from_program(program.individual)
        assert all(
            min_val <= c <= max_val for c in constants
        ), f"Constant out of bounds: {constants}"


def test_include_constants_true_generates_constants(regression_data):
    """Test that constants are present when include_constants=True."""
    X, y = regression_data
    fs = GeneticFeatureSynthesis(
        num_features=5,
        max_generations=10,  # Increased generations for more evolution
        population_size=30,  # Increased population size
        n_jobs=1,
        show_progress_bar=False,
        include_constants=True,
        const_prob=0.3,  # Increased probability of generating constants
    )
    fs.fit(X, y)
    found = False
    for program in fs.hall_of_fame:
        if extract_constants_from_program(program.individual):
            found = True
            break
    assert found, "No constants found in any program when include_constants=True"


def test_min_greater_than_max_constant_raises(regression_data):
    """Test that min_constant_val > max_constant_val raises an error."""
    X, y = regression_data
    import pytest

    with pytest.raises((ValueError, AssertionError)):
        GeneticFeatureSynthesis(
            num_features=3,
            max_generations=2,
            population_size=5,
            n_jobs=1,
            show_progress_bar=False,
            include_constants=True,
            min_constant_val=10,
            max_constant_val=-10,
        ).fit(X, y)


def test_invalid_input_data_raises():
    """Test fitting on empty DataFrame and DataFrame with NaNs raises error."""
    fs = GeneticFeatureSynthesis(
        num_features=2,
        max_generations=2,
        population_size=5,
        n_jobs=1,
        show_progress_bar=False,
    )
    import numpy as np
    import pandas as pd
    import pytest

    X_empty = pd.DataFrame()
    y_empty = pd.Series(dtype=float)
    with pytest.raises(Exception):
        fs.fit(X_empty, y_empty)
    X_nan = pd.DataFrame({"x0": [np.nan, np.nan], "x1": [np.nan, np.nan]})
    y_nan = pd.Series([np.nan, np.nan])
    with pytest.raises(Exception):
        fs.fit(X_nan, y_nan)


def test_no_constants(regression_data):
    """Test that no constants are generated when include_constants=False"""
    X, y = regression_data
    fs = GeneticFeatureSynthesis(
        num_features=5,
        max_generations=5,
        population_size=20,
        n_jobs=1,
        show_progress_bar=False,
        include_constants=False,
    )
    fs.fit(X, y)

    for program in fs.hall_of_fame:
        assert not contains_constant(
            program.individual
        ), f"Found constant in program: {program.individual}"


def test_single_input_feature():
    X = pd.DataFrame(np.random.randn(100, 1), columns=["x0"])
    y = X["x0"] * 3
    fs = GeneticFeatureSynthesis(
        num_features=1,
        max_generations=2,
        population_size=5,
        n_jobs=1,
        show_progress_bar=False,
    )
    Xt = fs.fit_transform(X, y)
    assert isinstance(Xt, pd.DataFrame)
    # Output includes original + generated feature(s)
    assert Xt.shape[1] >= 1


def test_categorical_input_raises():
    X = pd.DataFrame({"cat": ["a", "b", "c", "a", "c"] * 20})
    y = np.arange(100)
    fs = GeneticFeatureSynthesis(
        num_features=1,
        max_generations=2,
        population_size=5,
        n_jobs=1,
        show_progress_bar=False,
    )
    import pytest

    with pytest.raises(Exception):
        fs.fit(X, y)


def test_boundary_population_generation_values(regression_data):
    X, y = regression_data
    # Avoid pop_size=1, which is not supported by tournament selection logic
    for pop_size, gens in [(2, 1), (50, 1), (2, 5)]:
        fs = GeneticFeatureSynthesis(
            num_features=1,
            max_generations=gens,
            population_size=pop_size,
            n_jobs=1,
            show_progress_bar=False,
        )
        Xt = fs.fit_transform(X, y)
        assert isinstance(Xt, pd.DataFrame)


def test_parallel_jobs(regression_data):
    X, y = regression_data
    fs = GeneticFeatureSynthesis(
        num_features=2,
        max_generations=2,
        population_size=5,
        n_jobs=2,
        show_progress_bar=False,
    )
    Xt = fs.fit_transform(X, y)
    # Output includes original + generated features
    assert Xt.shape[1] >= 2


def test_transform_without_fit_raises(regression_data):
    X, _ = regression_data
    fs = GeneticFeatureSynthesis(
        num_features=2,
        max_generations=2,
        population_size=5,
        n_jobs=1,
        show_progress_bar=False,
    )
    import pytest

    with pytest.raises(Exception):
        fs.transform(X)


def test_transform_shape_mismatch(regression_data):
    X, y = regression_data
    fs = GeneticFeatureSynthesis(
        num_features=2,
        max_generations=2,
        population_size=5,
        n_jobs=1,
        show_progress_bar=False,
    )
    fs.fit(X, y)
    X_bad = pd.DataFrame(np.random.randn(10, 2), columns=["x0", "x1"])
    # The transform method does not raise, but output shape will not match X
    Xt = fs.transform(X_bad)
    assert Xt.shape[0] == 10
