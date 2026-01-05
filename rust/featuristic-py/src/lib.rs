//! Featuristic Python bindings
//!
//! This crate provides PyO3 bindings to the featuristic-core Rust library,
//! enabling high-performance symbolic regression and genetic programming
//! from Python.

use pyo3::prelude::*;

// Core modules
pub mod tree_bindings;
pub mod population_bindings;
pub mod mrmr_bindings;
pub mod binary_population_bindings;

/// Featuristic Python module
#[pymodule]
fn featuristic(_py: Python, m: &PyModule) -> PyResult<()> {
    // Tree operations
    m.add_function(wrap_pyfunction!(tree_bindings::evaluate_tree, m)?)?;
    m.add_function(wrap_pyfunction!(tree_bindings::random_tree, m)?)?;
    m.add_function(wrap_pyfunction!(tree_bindings::tree_to_string, m)?)?;
    m.add_function(wrap_pyfunction!(tree_bindings::tree_to_string_with_format, m)?)?;
    m.add_function(wrap_pyfunction!(tree_bindings::tree_depth, m)?)?;
    m.add_function(wrap_pyfunction!(tree_bindings::tree_node_count, m)?)?;

    // Population operations
    m.add_class::<population_bindings::Population>()?;

    // mRMR feature selection
    m.add_function(wrap_pyfunction!(mrmr_bindings::mrmr_select, m)?)?;
    m.add_class::<mrmr_bindings::MRMR>()?;

    // Binary population operations (feature selection)
    m.add_class::<binary_population_bindings::BinaryPopulation>()?;

    Ok(())
}
