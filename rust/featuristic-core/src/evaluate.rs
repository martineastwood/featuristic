//! Tree evaluation engine

use crate::tree::Node;
use ndarray::{Array1, ArrayView2};
use thiserror::Error;

/// Evaluation errors
#[derive(Debug, Error)]
pub enum EvalError {
    #[error("Missing feature: {0}")]
    MissingFeature(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(u32),

    #[error("Division by zero")]
    DivisionByZero,

    #[error("Invalid result (NaN or Inf)")]
    InvalidResult,
}

/// Stack-based evaluator for efficient tree traversal
pub struct Evaluator<'a> {
    feature_matrix: ArrayView2<'a, f64>,
}

impl<'a> Evaluator<'a> {
    /// Create a new evaluator
    pub fn new(feature_matrix: ArrayView2<'a, f64>) -> Self {
        Self { feature_matrix }
    }

    /// Evaluate a tree using stack-based traversal (avoids recursion)
    pub fn evaluate(&mut self, tree: &Node) -> Result<Array1<f64>, EvalError> {
        self.evaluate_node(tree)
    }

    /// Evaluate a single node
    fn evaluate_node(&self, node: &Node) -> Result<Array1<f64>, EvalError> {
        match node {
            // Constant leaf node
            Node::Constant(value) => {
                Ok(Array1::from_elem(self.feature_matrix.nrows(), *value))
            }

            // Feature leaf node
            Node::Feature { index, name } => {
                if *index >= self.feature_matrix.ncols() {
                    return Err(EvalError::MissingFeature(name.clone()));
                }
                let column = self.feature_matrix.column(*index);
                Ok(column.to_owned())
            }

            // Function node
            Node::Function { op_id, children, .. } => {
                // Recursively evaluate children (still using recursion for simplicity)
                // TODO: Convert to true stack-based traversal in optimization phase
                let child_results: Result<Vec<_>, _> = children
                    .iter()
                    .map(|child| self.evaluate_node(child))
                    .collect();
                let args = child_results?;

                // Apply operation
                apply_operation(*op_id, &args)
            }
        }
    }
}

/// Evaluate a symbolic tree node against a feature matrix
pub fn evaluate_tree(
    node: &Node,
    X: &ArrayView2<f64>,
) -> Result<Array1<f64>, EvalError> {
    match node {
        // Constant leaf node
        Node::Constant(value) => {
            Ok(Array1::from_elem(X.nrows(), *value))
        }

        // Feature leaf node
        Node::Feature { index, name } => {
            if *index >= X.ncols() {
                return Err(EvalError::MissingFeature(name.clone()));
            }
            let column = X.column(*index);
            Ok(column.to_owned())
        }

        // Function node
        Node::Function { op_id, children, .. } => {
            // Recursively evaluate children
            let child_results: Result<Vec<_>, _> = children
                .iter()
                .map(|child| evaluate_tree(child, X))
                .collect();
            let args = child_results?;

            // Apply operation
            apply_operation(*op_id, &args)
        }
    }
}

/// Apply a built-in operation to arguments
fn apply_operation(op_id: u32, args: &[Array1<f64>]) -> Result<Array1<f64>, EvalError> {
    use ndarray::Zip;

    let result = match op_id {
        // Binary operations
        0 => &args[0] + &args[1], // add
        1 => &args[0] - &args[1], // subtract
        2 => &args[0] * &args[1], // multiply
        3 => { // divide
            let result = &args[0] / &args[1];
            // Replace inf/nan with 1.0 (matching Python behavior)
            result.mapv(|x: f64| if x.is_finite() { x } else { 1.0 })
        }
        4 => { // min
            Zip::from(&args[0]).and(&args[1]).map_collect(|a, b| a.min(*b))
        }
        5 => { // max
            Zip::from(&args[0]).and(&args[1]).map_collect(|a, b| a.max(*b))
        }

        // Unary operations
        6 => args[0].mapv(|x: f64| x.sin()),   // sin
        7 => args[0].mapv(|x: f64| x.cos()),   // cos
        8 => { // tan (with clipping to prevent infinity at pi/2)
            args[0].mapv(|x: f64| {
                let clipped = x.clamp(-1.5, 1.5);  // Prevent approaching pi/2
                clipped.tan()
            })
        }
        9 => { // exp (with clipping)
            args[0].mapv(|x: f64| {
                let clipped = x.clamp(-20.0, 20.0);
                clipped.exp()
            })
        }
        10 => { // log (with error handling)
            args[0].mapv(|x: f64| {
                if x <= 0.0 {
                    0.0 // Replace invalid logs
                } else {
                    x.ln()
                }
            })
        }
        11 => { // sqrt
            args[0].mapv(|x: f64| {
                if x < 0.0 {
                    0.0 // Replace invalid sqrt
                } else {
                    x.sqrt()
                }
            })
        }
        12 => args[0].mapv(|x: f64| x.abs()),  // abs
        13 => -&args[0],                        // neg
        14 => &args[0] * &args[0],              // square
        15 => { // cube
            args[0].mapv(|x: f64| x.powi(3))
        }

        // Ternary operations
        16 => { // clip
            let min_val = args[1][0].min(args[2][0]);
            let max_val = args[1][0].max(args[2][0]);
            args[0].mapv(|x: f64| {
                x.clamp(min_val, max_val)
            })
        }

        _ => return Err(EvalError::InvalidOperation(op_id)),
    };

    // Check for invalid results - return large constant penalty
    // This ensures reproducibility and consistent strong penalty for invalid trees
    if result.iter().any(|x: &f64| !x.is_finite()) {
        // Return a large constant value (worse than any reasonable prediction)
        // MSE of 1e9 means predictions are off by ~31622 on average
        const INVALID_PENALTY: f64 = 1e9;
        Ok(Array1::from_elem(result.len(), INVALID_PENALTY))
    } else {
        Ok(result)
    }
}
