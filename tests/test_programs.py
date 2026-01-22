import featuristic as ft


def test_list_symbolic_functions():
    """Test that we can list all symbolic functions."""
    funcs = ft.synthesis.list_symbolic_functions()
    assert "add" in funcs
    assert "subtract" in funcs
    assert "multiply" in funcs
    assert "divide" in funcs


def test_symbolic_operation_metadata():
    """Test that we can get operation metadata from the constants."""
    # Test that we can access operation metadata through the constants
    from featuristic.constants import OP_KIND_METADATA, OP_NAME_TO_KIND

    # Check that some operations exist
    assert "add" in OP_NAME_TO_KIND
    assert "abs" in OP_NAME_TO_KIND

    # Check that we can get metadata
    add_kind = OP_NAME_TO_KIND["add"]
    assert add_kind in OP_KIND_METADATA

    name, fmt = OP_KIND_METADATA[add_kind]
    assert name == "add"
    assert fmt == "({} + {})"


def test_render_prog():
    """Test rendering a program to string."""
    # Create a simple program node using dict structure (how the deserializer works)
    node = {
        "kind": "add",
        "format_str": "({} + {})",
        "children": [
            {"kind": "feature", "feature_name": "a"},
            {"kind": "feature", "feature_name": "b"},
        ],
    }

    result = ft.synthesis.render_prog(node)
    assert result == "(a + b)"


def test_render_prog_nested():
    """Test rendering a nested program."""
    node = {
        "kind": "multiply",
        "format_str": "({} * {})",
        "children": [
            {"kind": "feature", "feature_name": "x"},
            {
                "kind": "add",
                "format_str": "({} + {})",
                "children": [
                    {"kind": "feature", "feature_name": "a"},
                    {"kind": "feature", "feature_name": "b"},
                ],
            },
        ],
    }

    result = ft.synthesis.render_prog(node)
    assert result == "(x * (a + b))"
