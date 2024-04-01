import featuristic as ft
import pandas as pd
import numpy as np


def test_random_prog():
    np.random.seed(0)
    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    operations = [
        ft.synthesis.symbolic_functions.SymbolicAdd,
        ft.synthesis.symbolic_functions.SymbolicSubtract,
    ]
    prog = ft.synthesis.program.random_prog(2, X, operations)

    assert isinstance(prog["func"], ft.synthesis.symbolic_functions.SymbolicAdd)
    assert "children" in prog
    assert "format_str" in prog
    assert "feature_name" in prog["children"][0]
    assert prog["children"][0]["feature_name"] in X.columns


def test_node_count():
    np.random.seed(0)
    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    node = {
        "func": ft.synthesis.symbolic_functions.SymbolicAdd(),
        "children": [
            {"feature_name": "a"},
            {"feature_name": "b"},
        ],
        "format_str": "({} + {})",
    }
    assert ft.synthesis.program.node_count(node) == 2


def test_select_random_node():
    np.random.seed(0)
    node = {
        "func": ft.synthesis.symbolic_functions.SymbolicAdd(),
        "children": [
            {"feature_name": "a"},
            {"feature_name": "b"},
        ],
        "format_str": "({} + {})",
    }
    selected = ft.synthesis.program.select_random_node(node, None, 2)

    assert type(selected) == dict
    assert type(selected["children"][0]) == dict
    assert "func" in selected


def test_render_prog():
    node = {
        "func": ft.synthesis.symbolic_functions.SymbolicAdd(),
        "children": [
            {"feature_name": "a"},
            {"feature_name": "b"},
        ],
        "format_str": "({} + {})",
    }

    assert ft.synthesis.program.render_prog(node) == "(a + b)"
