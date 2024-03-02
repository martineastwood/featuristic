import numpy as np


def random_prog(depth, X, operations):
    if np.random.randint(0, 10) >= depth * 2:
        op = operations[np.random.randint(0, len(operations) - 1)]
        return {
            "func": op,
            "children": [
                random_prog(depth + 1, X, operations) for _ in range(op.arg_count)
            ],
            "format_str": op.format_str,
        }
    else:
        return {"feature_name": X.columns[np.random.randint(0, X.shape[1] - 1)]}


def node_count(node):
    if "children" not in node:
        return 1
    return sum([node_count(c) for c in node["children"]])


def select_random_node(selected, parent, depth):
    if "children" not in selected:
        return parent

    if np.random.randint(0, 10) < 2 * depth:
        return selected

    child_count = len(selected["children"])
    child_idx = 0 if child_count <= 1 else np.random.randint(0, child_count - 1)

    return select_random_node(
        selected["children"][child_idx],
        selected,
        depth + 1,
    )


def render_prog(node):
    if "children" not in node:
        return node["feature_name"]
    return node["format_str"].format(*[render_prog(c) for c in node["children"]])
