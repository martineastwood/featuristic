from . import schema


def featurize(df):
    s = schema.infer_schema(df)

    return s
