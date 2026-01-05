//! PyO3 bindings for tree operations

use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray2, PyArray1, ToPyArray};
use featuristic_core::{Node, SymbolicTree, SymbolicOp};
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use pyo3::types::PyList;

/// Parse a Python dict into a Rust Node
fn parse_node_from_py(dict: &PyDict) -> PyResult<Node> {
    // Check for constant node
    if let Ok(value) = dict.get_item("value") {
        if let Some(val_obj) = value {
            let val = val_obj.extract::<f64>()?;
            return Ok(Node::Constant(val));
        }
    }

    // Check for feature node
    if let Ok(name) = dict.get_item("feature_name") {
        if let Some(name_obj) = name {
            let feature_name = name_obj.extract::<String>()?;
            // Try to get index from dict, default to 0 if not specified
            let index = if let Ok(idx_obj) = dict.get_item("index") {
                if let Some(idx) = idx_obj {
                    idx.extract::<usize>().unwrap_or(0)
                } else {
                    0
                }
            } else {
                0
            };
            return Ok(Node::Feature {
                name: feature_name,
                index,
            });
        }
    }

    // Function node - must have op_id, arity, and children
    let op_id_obj = dict.get_item("op_id")?;
    let op_id = if let Some(obj) = op_id_obj {
        obj.extract::<u32>()?
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing op_id"));
    };

    let arity_obj = dict.get_item("arity")?;
    let arity = if let Some(obj) = arity_obj {
        obj.extract::<u8>()?
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing arity"));
    };

    let children_obj = dict.get_item("children")?;
    let children_list: &PyList = if let Some(obj) = children_obj {
        obj.extract()?
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing children"));
    };

    let mut children = Vec::new();
    for child_dict in children_list.iter() {
        let child_dict: &PyDict = child_dict.downcast()?;
        children.push(parse_node_from_py(child_dict)?);
    }

    Ok(Node::Function {
        op_id,
        arity,
        children,
    })
}

/// Evaluate a symbolic tree on data.
///
/// Evaluates a symbolic tree (represented as a nested dictionary) on input data
/// and returns the predicted values.
///
/// # Arguments
/// * `tree_dict` - Tree represented as nested dictionary with keys:
///   - "function" (str): Function name or None for leaf nodes
///   - "children" (list): Child trees (for function nodes)
///   - "feature_name" (str): Feature name (for feature leaf nodes)
///   - "index" (int): Feature index (for feature leaf nodes)
///   - "value" (float): Constant value (for constant leaf nodes)
/// * `x` - Input data as 2D numpy array of shape (n_samples, n_features)
///
/// # Returns
/// Numpy array of shape (n_samples,) containing predicted values
///
/// # Example
/// ```python
/// import featuristic
/// import numpy as np
/// import pandas as pd
///
/// # Create a simple tree: x1 * x2
/// tree = {
///     'function': 'mul',
///     'children': [
///         {'feature_name': 'x1', 'index': 0, 'value': None},
///         {'feature_name': 'x2', 'index': 1, 'value': None}
///     ]
/// }
///
/// # Evaluate on data
/// X = pd.DataFrame({'x1': [1.0, 2.0, 3.0], 'x2': [4.0, 5.0, 6.0]})
/// result = featuristic.evaluate_tree(tree, X)
/// # result = array([4.0, 10.0, 18.0])
/// ```
#[pyfunction]
pub fn evaluate_tree(
    tree_dict: &PyDict,
    x: &PyArray2<f64>,
) -> PyResult<PyObject> {
    let node = parse_node_from_py(tree_dict)?;
    let tree = SymbolicTree::new(node);
    let x_view = unsafe { x.as_array() };

    let result = tree.evaluate(&x_view)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Evaluation error: {}", e)))?;

    Python::with_gil(|py| {
        Ok(result.to_pyarray(py).into())
    })
}

/// Generate a random symbolic tree for genetic programming.
///
/// Creates a random symbolic tree with the specified depth, using the given
/// feature names and built-in functions. The tree structure is generated
/// using probabilistic growth to create diverse shapes.
///
/// # Arguments
/// * `max_depth` - Maximum depth of the tree (recommended: 3-6)
/// * `feature_names` - List of input feature names
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// Dictionary representing the tree with keys:
/// * "expression" (str): Human-readable formula
/// * "depth" (int): Maximum depth of the tree
/// * "node_count" (int): Total number of nodes in the tree
///
/// # Example
/// ```python
/// import featuristic
///
/// tree = featuristic.random_tree(
///     max_depth=3,
///     feature_names=["x1", "x2", "x3"],
///     seed=42
/// )
/// # tree = {
/// #     'expression': '(x1 * x2) + x3',
/// #     'depth': 3,
/// #     'node_count': 5
/// # }
/// ```
#[pyfunction]
pub fn random_tree(
    max_depth: usize,
    feature_names: Vec<String>,
    seed: u64,
) -> PyResult<PyObject> {
    use featuristic_core::builtins::default_builtins;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let operations = default_builtins();

    let tree = SymbolicTree::random(
        max_depth,
        &feature_names,
        &operations,
        &mut rng,
        -10.0, 10.0, true, 0.15, 0.6,
    );

    Python::with_gil(|py| {
        let result = PyDict::new(py);
        result.set_item("depth", tree.get_depth())?;
        result.set_item("node_count", tree.node_count())?;
        result.set_item("expression", tree.to_string())?;
        Ok(result.into())
    })
}

/// Convert a symbolic tree to a human-readable string.
///
/// Converts the tree structure into a mathematical formula string using
/// standard function names (add, sub, mul, div, sin, cos, etc.).
///
/// # Arguments
/// * `tree_dict` - Tree represented as nested dictionary (see `evaluate_tree`)
///
/// # Returns
/// Human-readable formula string (e.g., "(x1 * x2) + sin(x3)")
///
/// # Example
/// ```python
/// import featuristic
///
/// tree = {
///     'function': 'add',
///     'children': [
///         {'function': 'mul', 'children': [...]},
///         {'function': 'sin', 'children': [...]}
///     ]
/// }
///
/// expr = featuristic.tree_to_string(tree)
/// # expr = "(x1 * x2) + sin(x3)"
/// ```
#[pyfunction]
pub fn tree_to_string(tree_dict: &PyDict) -> PyResult<String> {
    let node = parse_node_from_py(tree_dict)?;
    let tree = SymbolicTree::new(node);
    Ok(tree.to_string())
}

/// Convert a tree to string with custom feature formatting.
///
/// Similar to `tree_to_string` but allows custom format strings for features.
/// Useful for generating formula strings with specific feature notation.
///
/// # Arguments
/// * `tree_dict` - Tree represented as nested dictionary (see `evaluate_tree`)
/// * `format_strings` - List of format strings, one per feature
///   Use "{}" as placeholder for feature index or name
///
/// # Returns
/// Formula string with custom feature formatting
///
/// # Example
/// ```python
/// import featuristic
///
/// tree = {...}  # Tree using features 0, 1, 2
///
/// # Use Feature() notation
/// expr = featuristic.tree_to_string_with_format(
///     tree,
///     format_strings=["Feature({})", "Feature({})", "Feature({})"]
/// )
/// # expr = "(Feature(0) * Feature(1)) + Feature(2)"
///
/// # Use x[] notation
/// expr = featuristic.tree_to_string_with_format(
///     tree,
///     format_strings=["x[{}]", "x[{}]", "x[{}]"]
/// )
/// # expr = "(x[0] * x[1]) + x[2]"
/// ```
#[pyfunction]
pub fn tree_to_string_with_format(tree_dict: &PyDict, format_strings: Vec<String>) -> PyResult<String> {
    let node = parse_node_from_py(tree_dict)?;
    let tree = SymbolicTree::new(node);
    Ok(tree.to_string_with_format(&format_strings))
}

/// Get the depth of a symbolic tree.
///
/// Returns the maximum depth from root to any leaf node in the tree.
/// Depth is a measure of tree complexity and nesting level.
///
/// # Arguments
/// * `tree_dict` - Tree represented as nested dictionary (see `evaluate_tree`)
///
/// # Returns
/// Maximum depth of the tree (root at depth 0)
///
/// # Example
/// ```python
/// import featuristic
///
/// # Tree: (x1 + x2) * x3
/// # Structure:
/// #   mul (depth 0)
/// #     ├── add (depth 1)
/// #     │   ├── x1 (depth 2)
/// #     │   └── x2 (depth 2)
/// #     └── x3 (depth 1)
///
/// depth = featuristic.tree_depth(tree)
/// # depth = 2
/// ```
#[pyfunction]
pub fn tree_depth(tree_dict: &PyDict) -> PyResult<usize> {
    let node = parse_node_from_py(tree_dict)?;
    let tree = SymbolicTree::new(node);
    Ok(tree.get_depth())
}

/// Count the number of nodes in a symbolic tree.
///
/// Returns the total number of nodes (functions + features + constants) in the tree.
/// Optionally applies a weighting factor to constant nodes.
///
/// # Arguments
/// * `tree_dict` - Tree represented as nested dictionary (see `evaluate_tree`)
/// * `const_weight` - Optional weight for constant nodes (default: 1.0)
///   Set to 0.0 to ignore constants, higher values penalize them more
///
/// # Returns
/// Total node count (or weighted count if const_weight is provided)
///
/// # Example
/// ```python
/// import featuristic
///
/// # Tree: (x1 + x2) * x3 + 5.0
/// # Nodes: mul, add, x1, x2, x3, 5.0 (6 nodes total)
///
/// count = featuristic.tree_node_count(tree)
/// # count = 6.0
///
/// # Weighted count (constants don't count as much)
/// weighted = featuristic.tree_node_count(tree, const_weight=0.1)
/// # weighted = 5.5 (5 regular nodes + 0.1 × 1 constant)
/// ```
#[pyfunction]
pub fn tree_node_count(tree_dict: &PyDict, const_weight: Option<f64>) -> PyResult<f64> {
    let node = parse_node_from_py(tree_dict)?;
    let tree = SymbolicTree::new(node);

    if let Some(w) = const_weight {
        Ok(tree.weighted_node_count(w))
    } else {
        Ok(tree.node_count() as f64)
    }
}
