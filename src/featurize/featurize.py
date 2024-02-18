import featurize as ft
from featurize import schema
import pandas as pd
import itertools


def featurize(df):
    s = schema.Schema()
    s.infer_schema(df)

    df = _add_numerical_features(df, s)

    df = ft.selection.utils.remove_low_quality_features(df)

    s.infer_schema(df)

    df = _add_combination_features(df, s)

    s.infer_schema(df)

    df = ft.selection.utils.remove_low_quality_features(df)

    return df


def _add_numerical_features(df, schema):
    l_data = []
    l_label = []

    for col in schema.get_numerical_columns():
        for _, v in ft.transformations.numeric.transfomers.items():
            l_data.append(v(df[col]))
            l_label.append(v.get_column_names(col))

    df_append = pd.DataFrame(zip(*l_data), index=df.index, columns=l_label)
    df = pd.concat([df, df_append], axis=1)

    return df


def _add_combination_features(df, schema):
    l_data = []
    l_label = []

    pairs = list(itertools.combinations(schema.get_numerical_columns(), 2))
    for pair in pairs:
        for _, v in ft.transformations.combinations.transformers.items():
            l_data.append(v(df[pair[0]], df[pair[1]]))
            l_label.append(v.get_column_names(pair[0], pair[1]))

    df_append = pd.DataFrame(zip(*l_data), index=df.index, columns=l_label)
    df = pd.concat([df, df_append], axis=1)

    return df
