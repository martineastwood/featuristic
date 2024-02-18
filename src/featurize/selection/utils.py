def remove_zero_variance_columns(df):
    """
    Remove features that are not useful for modeling
    """
    return df.loc[:, df.nunique() != 1]
