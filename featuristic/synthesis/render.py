"""Utility functions for rendering and simplifying programs."""


def simplify_program(node: dict, depth: int = 0, max_depth: int = 10) -> dict:
    """
    Simplify a program by removing redundant operations.

    Reduces:
    - Double negations: negate(negate(x)) → x
    - Triple negations: negate(negate(negate(x))) → negate(x)
    - Identity operations: x + 0 → x, x * 1 → x, x - 0 → x
    - Zero operations: x * 0 → 0, 0 / x → 0, x - x → 0

    Parameters
    ----------
    node : dict
        The program node to simplify
    depth : int
        Current recursion depth (for safety)
    max_depth : int
        Maximum recursion depth to prevent infinite loops

    Returns
    -------
    dict
        Simplified program node
    """
    if depth > max_depth:
        return node

    if "children" not in node:
        # Leaf node - nothing to simplify
        return node

    # First, recursively simplify all children
    simplified_children = [
        simplify_program(c, depth + 1, max_depth) for c in node["children"]
    ]

    # Get operation type from format_str
    format_str = node.get("format_str", "")

    # Helper to check if a child is a constant
    def is_constant(n: dict, value: float = None) -> bool:
        if "children" in n:
            return False
        try:
            name = n.get("feature_name", "")
            if value is not None:
                return float(name) == value
            return name.replace(".", "", 1).replace(
                "-", "", 1
            ).isdigit() or name.startswith("-")
        except:
            return False

    def get_constant_value(n: dict) -> float:
        try:
            return float(n.get("feature_name", 0))
        except:
            return 0.0

    # Apply simplification rules
    if format_str == "negate({})":
        child = simplified_children[0]
        child_format = child.get("format_str", "")

        # Double negation: negate(negate(x)) → x
        if child_format == "negate({})" and "children" in child:
            return child["children"][0]  # Return grandchild

        # Triple negation: negate(negate(negate(x))) → negate(x)
        if child_format == "negate({})":
            grandchild = (
                child.get("children", [None])[0] if child.get("children") else None
            )
            if grandchild and grandchild.get("format_str") == "negate({})":
                return grandchild["children"][0]  # Return great-grandchild

        # Single negation - keep it
        return {**node, "children": simplified_children}

    elif format_str == "({} + {})":
        left, right = simplified_children

        # x + 0 → x
        if is_constant(right, 0.0):
            return left
        if is_constant(left, 0.0):
            return right

        return {**node, "children": simplified_children}

    elif format_str == "({} - {})":
        left, right = simplified_children

        # x - 0 → x
        if is_constant(right, 0.0):
            return left

        # x - x → 0
        if left == right:
            return {"feature_name": "0.0"}

        return {**node, "children": simplified_children}

    elif format_str == "({} * {})":
        left, right = simplified_children

        # x * 0 → 0
        if is_constant(left, 0.0) or is_constant(right, 0.0):
            return {"feature_name": "0.0"}

        # x * 1 → x
        if is_constant(right, 1.0):
            return left
        if is_constant(left, 1.0):
            return right

        return {**node, "children": simplified_children}

    elif format_str == "({} / {})":
        left, right = simplified_children

        # 0 / x → 0
        if is_constant(left, 0.0):
            return {"feature_name": "0.0"}

        # x / 1 → x
        if is_constant(right, 1.0):
            return left

        return {**node, "children": simplified_children}

    elif format_str == "square({})":
        child = simplified_children[0]

        # square(square(x)) - could become x^4, but we'll leave it
        # square(negate(x)) - leave as is
        return {**node, "children": simplified_children}

    elif format_str == "cube({})":
        child = simplified_children[0]

        # cube(cube(x)) - leave as is
        return {**node, "children": simplified_children}

    # Default: no simplification applied
    return {**node, "children": simplified_children}


def render_prog(node: dict, simplify: bool = True) -> str:
    """
    Render a program to a string representation.

    This function converts a deserialized program (from Nim) into a
    human-readable formula string.

    Parameters
    ----------
    node : dict
        The program node to render. Should have either:
        - "feature_name" key for leaf nodes
        - "format_str" and "children" keys for internal nodes
    simplify : bool
        Whether to simplify the program before rendering (default: True)

    Returns
    -------
    str
        The rendered formula string
    """
    if simplify:
        node = simplify_program(node)

    if "children" not in node:
        # Leaf node
        return node["feature_name"]

    # Internal node - use format_str to render children
    return node["format_str"].format(
        *[render_prog(c, simplify=False) for c in node["children"]]
    )
