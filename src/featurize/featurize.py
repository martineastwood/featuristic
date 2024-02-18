import featurize as ft
from featurize import schema
import pandas as pd
import itertools
from typing import Callable, Union
from functools import partial
from featurize.logging import logger


def featurize(
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str = "regression",
    feature_depth: int = 1,
    mrmr_k: int = None,
    swarm_particles: int = 10,
    swarm_iters: int = 10,
    cost_function: Union[None, Callable] = None,
):
    """
    Featurize a dataframe.

    Parameters
    ----------

    df : pandas.DataFrame
        The dataframe to featurize.

    problem_type : str
        The type of problem to solve. Either "regression" or "classification".

    feature_depth : int
        The number of steps to take when featurizing the dataframe.

    mrmr_k : int
        The number of features to select using the MRMR algorithm.
    """

    logger.info(
        "Checking arguments to featurize function are in within acceptable bounds"
    )
    if problem_type not in ["regression", "classification"]:
        raise ValueError(
            f"Problem_type must be either 'regression' or 'classification'. Got {problem_type}"
        )

    if mrmr_k is None:
        mrmr_k = int(X.shape[0] * 0.5)
    elif mrmr_k < 1:
        raise ValueError("mrmr_k must be greater than 0")

    if feature_depth < 1:
        raise ValueError("steps must be greater than 0")

    if cost_function is None:
        if problem_type == "regression":
            cost_function = ft.cost_funcs.regression.linear_reg_mae

    logger.info("Inferring initial dataframe schema")
    s = schema.Schema()
    s.infer_schema(X)

    for i in range(feature_depth):
        logger.info(f"Featurizing dataframe at depth {i + 1}")
        X, s = _featurize(X, s, i)

    logger.info("Selecting features using MRMR algorithm")
    mrmr = ft.selection.MaxRelevanceMinRedundancy(K=mrmr_k)
    X = mrmr.fit_transform(X, y)

    logger.info(
        f"Optimizing features using Particle Swarm Optimizer with {swarm_particles} particles and {X.shape[1]} dimensions."
    )
    pso = ft.selection.BinaryParticleSwarmOptimiser(
        num_particles=swarm_particles, num_dimensions=X.shape[1]
    )

    logger.info("Creating partial function for cost function")
    f = partial(cost_function, X=X, y=y)

    cost, position = pso.optimize(f, max_iters=10)
    logger.info(
        f"Optimization completed with final cost: {cost} and {position.sum()} features selected."
    )

    return X[X.columns[position == 1]]


def _featurize(df, schema: schema.Schema, step: int):
    """
    Featurize the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to featurize.

    schema : featurize.schema.Schema
        The schema of the dataframe.

    step : int
        The step in the featurization process.
    """
    logger.info("Adding numerical features")
    df = _add_numerical_features(df, schema, step)

    old_shape = df.shape
    df = ft.selection.utils.remove_zero_variance_columns(df)
    logger.info(f"Removed {old_shape[1] - df.shape[1]} zero variance columns")
    schema.infer_schema(df)

    logger.info("Adding combination features")
    df = _add_combination_features(df, schema, step)

    old_shape = df.shape
    df = ft.selection.utils.remove_zero_variance_columns(df)
    logger.info(f"Removed {old_shape[1] - df.shape[1]} zero variance columns")

    logger.info("Infering schema")
    schema.infer_schema(df)
    return df, schema


def _add_numerical_features(df, schema, step):
    """
    Add numerical features to the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to featurize.

    schema : featurize.schema.Schema
        The schema of the dataframe.

    step : int
        The step in the featurization process.
    """
    l_data = []
    l_label = []

    for col in schema.get_numerical_columns():
        for _, v in ft.transformations.numeric.transfomers.items():
            l_data.append(v(df[col]))
            l_label.append(v.get_column_names(col) + f"_{step}")

    df_append = pd.DataFrame(zip(*l_data), index=df.index, columns=l_label)
    df = pd.concat([df, df_append], axis=1)

    return df


def _add_combination_features(df, schema, step):
    """
    Add combination features to the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to featurize.

    schema : featurize.schema.Schema
        The schema of the dataframe.

    step : int
        The step in the featurization process.
    """
    l_data = []
    l_label = []

    pairs = list(itertools.combinations(schema.get_numerical_columns(), 2))
    for pair in pairs:
        for _, v in ft.transformations.combinations.transformers.items():
            l_data.append(v(df[pair[0]], df[pair[1]]))
            l_label.append(v.get_column_names(pair[0], pair[1]) + f"_{step}")

    df_append = pd.DataFrame(zip(*l_data), index=df.index, columns=l_label)
    df = pd.concat([df, df_append], axis=1)

    return df
