def remove_low_quality_features(df):
    """
    Remove features that are not useful for modeling
    """
    return df.loc[:, df.nunique() != 1]
