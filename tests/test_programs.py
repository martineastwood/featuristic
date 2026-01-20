import featuristic as ft
import pandas as pd
import numpy as np


def test_list_symbolic_functions():
    """Test that we can list all symbolic functions."""
    funcs = ft.synthesis.list_symbolic_functions()
    assert "add" in funcs
    assert "subtract" in funcs
    assert "multiply" in funcs
    assert "divide" in funcs


def test_symbolic_operation_metadata():
    """Test that symbolic operations derive metadata from constants."""
    add_op = ft.synthesis.symbolic_functions.SymbolicAdd()
    assert add_op.name == "add"
    assert add_op.format_str == "({} + {})"
    assert add_op.arg_count == 2

    abs_op = ft.synthesis.symbolic_functions.SymbolicAbs()
    assert abs_op.name == "abs"
    assert abs_op.format_str == "abs({})"
    assert abs_op.arg_count == 1


def test_render_prog():
    """Test rendering a program to string."""
    node = {
        "func": ft.synthesis.symbolic_functions.SymbolicAdd(),
        "children": [
            {"feature_name": "a"},
            {"feature_name": "b"},
        ],
        "format_str": "({} + {})",
    }

    assert ft.synthesis.render_prog(node) == "(a + b)"


def test_render_prog_nested():
    """Test rendering a nested program."""
    node = {
        "func": ft.synthesis.symbolic_functions.SymbolicMultiply(),
        "children": [
            {"feature_name": "x"},
            {
                "func": ft.synthesis.symbolic_functions.SymbolicAdd(),
                "children": [
                    {"feature_name": "a"},
                    {"feature_name": "b"},
                ],
                "format_str": "({} + {})",
            },
        ],
        "format_str": "({} * {})",
    }

    assert ft.synthesis.render_prog(node) == "(x * (a + b))"
