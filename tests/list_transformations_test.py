import featurize as ft


def test_list_transformations():
    transformations = ft.list_transformations()
    assert transformations.shape[0] > 0
    assert transformations.shape[1] == 3
    assert "name" in transformations.columns
    assert "type" in transformations.columns
    assert "description" in transformations.columns
    assert transformations["type"].isin(["numeric", "combinations"]).all()
    assert transformations["description"].apply(lambda x: len(x) > 0).all()
    assert transformations["name"].apply(lambda x: len(x) > 0).all()
    assert transformations["name"].duplicated().sum() == 0
    assert transformations["description"].duplicated().sum() == 0
