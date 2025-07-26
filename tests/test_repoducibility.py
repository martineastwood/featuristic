import numpy as np
import pandas as pd
import pytest

from featuristic.core.binary_population import BinaryPopulation
from featuristic import FeatureSynthesis
from featuristic.core.mrmr import MaxRelevanceMinRedundancy

SEED = 42


def set_seed(seed):
    np.random.seed(seed)


def dummy_cost(X, y):
    return X.sum().sum()


@pytest.fixture
def setup_population():
    def _setup(seed):
        set_seed(seed)
        pop = BinaryPopulation(population_size=10, feature_count=5)
        return pop

    return _setup


def test_binary_population_reproducibility(setup_population):
    # Setup run 1
    pop1 = setup_population(SEED)
    X_dummy = pd.DataFrame(np.random.rand(10, 5))
    y_dummy = pd.Series(np.random.randint(0, 2, size=10))

    fitness1 = pop1.evaluate(dummy_cost, X_dummy, y_dummy)
    pop1.evolve(fitness1)
    evolved1 = np.array(pop1.population)

    # Setup run 2
    pop2 = setup_population(SEED)
    X_dummy2 = pd.DataFrame(np.random.rand(10, 5))
    y_dummy2 = pd.Series(np.random.randint(0, 2, size=10))

    fitness2 = pop2.evaluate(dummy_cost, X_dummy2, y_dummy2)
    pop2.evolve(fitness2)
    evolved2 = np.array(pop2.population)

    np.testing.assert_array_equal(evolved1, evolved2)


def create_dummy_data():
    set_seed(SEED)
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.random.rand(100))
    return X, y


def test_feature_synthesis_reproducibility():
    X1, y1 = create_dummy_data()
    X2, y2 = create_dummy_data()

    set_seed(SEED)
    fs1 = FeatureSynthesis(
        num_features=3,
        population_size=20,
        max_generations=5,
        n_jobs=1,
        pbar=False,
    )
    out1 = fs1.fit_transform(X1, y1)

    set_seed(SEED)
    fs2 = FeatureSynthesis(
        num_features=3,
        population_size=20,
        max_generations=5,
        n_jobs=1,
        pbar=False,
    )
    out2 = fs2.fit_transform(X2, y2)

    pd.testing.assert_frame_equal(out1, out2)

    info1 = fs1.get_feature_info()
    info2 = fs2.get_feature_info()
    pd.testing.assert_frame_equal(info1, info2)


def create_dummy_data(n_samples=100, n_features=10):
    set_seed(SEED)
    X = pd.DataFrame(
        np.random.rand(n_samples, n_features),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.rand(n_samples))
    return X, y


def test_mrmr_reproducibility():
    X1, y1 = create_dummy_data()
    X2, y2 = create_dummy_data()

    set_seed(SEED)
    mrmr1 = MaxRelevanceMinRedundancy(k=5, pbar=False)
    mrmr1.fit(X1, y1)
    selected1 = mrmr1.selected_features.copy()

    set_seed(SEED)
    mrmr2 = MaxRelevanceMinRedundancy(k=5, pbar=False)
    mrmr2.fit(X2, y2)
    selected2 = mrmr2.selected_features.copy()

    assert selected1 == selected2


def create_dummy_data_selection(n_samples=100, n_features=6):
    set_seed(SEED)
    X = pd.DataFrame(
        np.random.rand(n_samples, n_features),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.rand(n_samples))
    return X, y


def test_feature_selection_reproducibility():
    X1, y1 = create_dummy_data_selection()
    X2, y2 = create_dummy_data_selection()

    set_seed(SEED)
    fs1 = FeatureSynthesis(
        num_features=3,
        population_size=20,
        max_generations=5,
        tournament_size=5,
        crossover_proba=0.8,
        parsimony_coefficient=0.001,
        n_jobs=1,
        pbar=False,
    )
    fs1.fit(X1, y1)
    info1 = fs1.get_feature_info()

    set_seed(SEED)
    fs2 = FeatureSynthesis(
        num_features=3,
        population_size=20,
        max_generations=5,
        tournament_size=5,
        crossover_proba=0.8,
        parsimony_coefficient=0.001,
        n_jobs=1,
        pbar=False,
    )
    fs2.fit(X2, y2)
    info2 = fs2.get_feature_info()

    pd.testing.assert_frame_equal(info1, info2)
    assert list(info1["name"]) == list(info2["name"])
    assert list(info1["formula"]) == list(info2["formula"])
