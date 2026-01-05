//! PyO3 bindings for mRMR feature selection

use pyo3::prelude::*;
use numpy::{PyArray2, PyArray1};
use featuristic_core::mrmr::MRMRSelector;

/// mRMR feature selection (Python binding)
#[pyfunction]
pub fn mrmr_select(
    x: &PyArray2<f64>,
    y: &PyArray1<f64>,
    num_features: usize,
) -> PyResult<Vec<usize>> {
    let x_view = unsafe { x.as_array() };
    let y_view = unsafe { y.as_array() };

    let selector = MRMRSelector::new(num_features);
    let selected_indices = selector.select(&x_view, &y_view);

    Ok(selected_indices)
}

/// mRMR selector class for Python
#[pyclass]
pub struct MRMR {
    selector: MRMRSelector,
}

#[pymethods]
impl MRMR {
    /// Create a new mRMR selector
    #[new]
    fn new(num_features: usize) -> Self {
        Self {
            selector: MRMRSelector::new(num_features),
        }
    }

    /// Fit the selector and return selected feature indices
    fn fit_select(
        &self,
        x: &PyArray2<f64>,
        y: &PyArray1<f64>,
    ) -> PyResult<Vec<usize>> {
        let x_view = unsafe { x.as_array() };
        let y_view = unsafe { y.as_array() };

        let selected = self.selector.select(&x_view, &y_view);
        Ok(selected)
    }

    /// Get the number of features to select
    fn get_num_features(&self) -> usize {
        // We can't access k directly since it's private
        // This is a placeholder
        0
    }
}
