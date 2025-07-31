import numpy as np
import pandas as pd
import pytest

from featuristic.core.binary_population import BinaryPopulation
from featuristic import GeneticFeatureSynthesis
from featuristic.core.mrmr import MaxRelevanceMinRedundancy
from featuristic.engine.feature_selector import GeneticFeatureSelector

SEED = 42


def dummy_cost(X, y):
    return X.sum().sum()


@pytest.fixture
def setup_population():
    def _setup(seed):
        rng = np.random.default_rng(seed)
        pop = BinaryPopulation(population_size=10, feature_count=5, rng=rng)
        return pop

    return _setup


def test_binary_population_reproducibility(setup_population):
    # Setup run 1
    rng1 = np.random.default_rng(SEED)
    pop1 = setup_population(SEED)
    X_dummy = pd.DataFrame(rng1.random(size=(10, 5)))
    y_dummy = pd.Series(rng1.integers(0, 2, size=10))

    fitness1 = pop1.evaluate(dummy_cost, X_dummy, y_dummy)
    pop1.evolve(fitness1)
    evolved1 = np.array(pop1.population)

    # Setup run 2
    rng2 = np.random.default_rng(SEED)
    pop2 = setup_population(SEED)
    X_dummy2 = pd.DataFrame(rng2.random(size=(10, 5)))
    y_dummy2 = pd.Series(rng2.integers(0, 2, size=10))

    fitness2 = pop2.evaluate(dummy_cost, X_dummy2, y_dummy2)
    pop2.evolve(fitness2)
    evolved2 = np.array(pop2.population)

    np.testing.assert_array_equal(evolved1, evolved2)


def create_dummy_data(seed, n_samples=100, n_features=5):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.random(size=(n_samples, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(rng.random(size=n_samples))
    return X, y


def test_feature_synthesis_reproducibility():
    X1, y1 = create_dummy_data(SEED)
    X2, y2 = create_dummy_data(SEED)

    rng1 = np.random.default_rng(SEED)
    fs1 = GeneticFeatureSynthesis(
        num_features=3,
        population_size=20,
        max_generations=5,
        n_jobs=1,
        show_progress_bar=False,
        rng=rng1,
    )
    out1 = fs1.fit_transform(X1, y1)

    rng2 = np.random.default_rng(SEED)
    fs2 = GeneticFeatureSynthesis(
        num_features=3,
        population_size=20,
        max_generations=5,
        n_jobs=1,
        show_progress_bar=False,
        rng=rng2,
    )
    out2 = fs2.fit_transform(X2, y2)

    pd.testing.assert_frame_equal(out1, out2)

    info1 = fs1.get_feature_info()
    info2 = fs2.get_feature_info()
    pd.testing.assert_frame_equal(info1, info2)


def create_dummy_data_mrmr(seed, n_samples=100, n_features=10):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.random(size=(n_samples, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(rng.random(size=n_samples))
    return X, y


def test_mrmr_reproducibility():
    X1, y1 = create_dummy_data_mrmr(SEED)
    X2, y2 = create_dummy_data_mrmr(SEED)

    rng1 = np.random.default_rng(SEED)
    mrmr1 = MaxRelevanceMinRedundancy(k=5, show_progress_bar=False, rng=rng1)
    mrmr1.fit(X1, y1)
    selected1 = mrmr1.selected_features_.copy()

    rng2 = np.random.default_rng(SEED)
    mrmr2 = MaxRelevanceMinRedundancy(k=5, show_progress_bar=False, rng=rng2)
    mrmr2.fit(X2, y2)
    selected2 = mrmr2.selected_features_.copy()

    assert selected1 == selected2


def create_dummy_data_selection(seed, n_samples=100, n_features=6):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.random(size=(n_samples, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(rng.random(size=n_samples))
    return X, y


def test_feature_selection_reproducibility():
    X1, y1 = create_dummy_data_selection(SEED)
    X2, y2 = create_dummy_data_selection(SEED)

    rng1 = np.random.default_rng(SEED)
    fs1 = GeneticFeatureSelector(
        objective_function=dummy_cost,
        population_size=20,
        max_generations=5,
        tournament_size=5,
        crossover_proba=0.8,
        n_jobs=1,
        show_progress_bar=False,
        rng=rng1,
    )
    fs1.fit(X1, y1)
    # GeneticFeatureSelector does not have get_feature_info, so we check selected_columns_
    selected_cols1 = fs1.selected_columns_.tolist()

    rng2 = np.random.default_rng(SEED)
    fs2 = GeneticFeatureSelector(
        objective_function=dummy_cost,
        population_size=20,
        max_generations=5,
        tournament_size=5,
        crossover_proba=0.8,
        n_jobs=1,
        show_progress_bar=False,
        rng=rng2,
    )
    fs2.fit(X2, y2)
    selected_cols2 = fs2.selected_columns_.tolist()

    assert selected_cols1 == selected_cols2
