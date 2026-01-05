//! Symbolic tree data structures and operations

use serde::{Deserialize, Serialize};
use std::fmt;
use rand::Rng;
use ndarray::ArrayView2;
use crate::evaluate::{Evaluator, EvalError};

/// Symbolic operation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolicOp {
    pub name: String,
    pub arity: u8,
    pub format_str: String,
    pub op_id: u32,
}

/// Symbolic tree node (enum-based for type safety)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Node {
    /// Leaf node: feature reference
    Feature { name: String, index: usize },
    /// Leaf node: constant value
    Constant(f64),
    /// Internal node: function application
    Function {
        op_id: u32,
        arity: u8,
        children: Vec<Node>,
    },
}

/// Main symbolic tree structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolicTree {
    pub root: Node,
    pub depth: usize,
}

impl SymbolicTree {
    /// Create a new symbolic tree
    pub fn new(root: Node) -> Self {
        let depth = Self::calculate_depth(&root);
        Self { root, depth }
    }

    /// Calculate the depth of a tree
    pub fn calculate_depth(node: &Node) -> usize {
        match node {
            Node::Feature { .. } | Node::Constant(_) => 1,
            Node::Function { children, .. } => {
                1 + children.iter().map(Self::calculate_depth).max().unwrap_or(0)
            }
        }
    }

    /// Get the depth of the tree
    pub fn get_depth(&self) -> usize {
        self.depth
    }

    /// Count the number of nodes in the tree
    pub fn node_count(&self) -> usize {
        self.count_nodes(&self.root)
    }

    fn count_nodes(&self, node: &Node) -> usize {
        match node {
            Node::Feature { .. } | Node::Constant(_) => 1,
            Node::Function { children, .. } => {
                1 + children.iter().map(|c| self.count_nodes(c)).sum::<usize>()
            }
        }
    }

    /// Count nodes with heavier weight for constants (for parsimony penalty)
    pub fn weighted_node_count(&self, const_weight: f64) -> f64 {
        self.count_nodes_weighted(&self.root, const_weight)
    }

    fn count_nodes_weighted(&self, node: &Node, const_weight: f64) -> f64 {
        match node {
            Node::Constant(_) => const_weight,
            Node::Feature { .. } => 1.0,
            Node::Function { children, .. } => {
                1.0 + children.iter().map(|c| self.count_nodes_weighted(c, const_weight)).sum::<f64>()
            }
        }
    }

    /// Generate a random symbolic tree using ramped half-and-half initialization
    pub fn random<R: Rng>(
        max_depth: usize,
        feature_names: &[String],
        operations: &[SymbolicOp],
        rng: &mut R,
        min_constant_val: f64,
        max_constant_val: f64,
        include_constants: bool,
        const_prob: f64,
        stop_prob: f64,
    ) -> Self {
        let root = Self::random_node(
            0,
            max_depth,
            feature_names,
            operations,
            rng,
            min_constant_val,
            max_constant_val,
            include_constants,
            const_prob,
            stop_prob,
        );
        Self::new(root)
    }

    /// Recursively generate a random node
    fn random_node<R: Rng>(
        depth: usize,
        max_depth: usize,
        feature_names: &[String],
        operations: &[SymbolicOp],
        rng: &mut R,
        min_constant_val: f64,
        max_constant_val: f64,
        include_constants: bool,
        const_prob: f64,
        stop_prob: f64,
    ) -> Node {
        // Decide whether to make a leaf
        let should_stop = depth >= max_depth || rng.gen::<f64>() < stop_prob;

        if should_stop {
            // Create a leaf node
            Self::random_leaf(
                feature_names,
                rng,
                min_constant_val,
                max_constant_val,
                include_constants,
                const_prob,
            )
        } else {
            // Create a function node
            let op = operations[rng.gen_range(0..operations.len())].clone();
            let mut children = Vec::with_capacity(op.arity as usize);

            for _ in 0..op.arity {
                children.push(Self::random_node(
                    depth + 1,
                    max_depth,
                    feature_names,
                    operations,
                    rng,
                    min_constant_val,
                    max_constant_val,
                    include_constants,
                    const_prob,
                    stop_prob,
                ));
            }

            Node::Function {
                op_id: op.op_id,
                arity: op.arity,
                children,
            }
        }
    }

    /// Generate a random leaf node (feature or constant)
    fn random_leaf<R: Rng>(
        feature_names: &[String],
        rng: &mut R,
        min_constant_val: f64,
        max_constant_val: f64,
        include_constants: bool,
        const_prob: f64,
    ) -> Node {
        // If no features, force a constant
        if feature_names.is_empty() {
            if include_constants {
                return Node::Constant(rng.gen_range(min_constant_val..=max_constant_val));
            } else {
                panic!("No features available and constants disabled");
            }
        }

        // Decide between constant and feature
        let use_constant = include_constants && rng.gen::<f64>() < const_prob;

        if use_constant {
            Node::Constant(rng.gen_range(min_constant_val..=max_constant_val))
        } else {
            let index = rng.gen_range(0..feature_names.len());
            Node::Feature {
                name: feature_names[index].clone(),
                index,
            }
        }
    }

    /// Select a random node from the tree (for mutation/crossover)
    pub fn select_random_node<R: Rng>(&self, rng: &mut R) -> &Node {
        self.select_random_node_recursive(&self.root, 0, rng)
    }

    fn select_random_node_recursive<'a, R: Rng>(
        &'a self,
        node: &'a Node,
        depth: usize,
        rng: &mut R,
    ) -> &'a Node {
        match node {
            Node::Feature { .. } | Node::Constant(_) => node,
            Node::Function { children, .. } => {
                // Bias towards deeper nodes (similar to Python implementation)
                if !children.is_empty() && rng.gen_range(0..10) < 2 * depth {
                    node
                } else {
                    let child_idx = rng.gen_range(0..children.len());
                    self.select_random_node_recursive(&children[child_idx], depth + 1, rng)
                }
            }
        }
    }

    /// Get a mutable reference to a randomly selected node
    /// Returns the path to the node and a mutable reference
    pub fn select_random_node_mut<R: Rng>(&mut self, _rng: &mut R) -> Vec<usize> {
        // TODO: Implement path tracking for mutation
        // For now, return empty path
        vec![]
    }
}

impl fmt::Display for SymbolicTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.render_node(&self.root))
    }
}

impl SymbolicTree {
    fn render_node(&self, node: &Node) -> String {
        match node {
            Node::Feature { name, .. } => name.clone(),
            Node::Constant(val) => format!("{:.3}", val),
            Node::Function { op_id, children, .. } => {
                // For now, use a generic format
                let child_strs: Vec<String> = children
                    .iter()
                    .map(|c| self.render_node(c))
                    .collect();
                format!("op_{}({})", op_id, child_strs.join(", "))
            }
        }
    }

    /// Simplify the tree expression (placeholder for future SymPy integration)
    pub fn simplify(&self) -> String {
        self.to_string()
    }

    /// Evaluate the tree against feature data
    pub fn evaluate(&self, X: &ArrayView2<f64>) -> Result<ndarray::Array1<f64>, EvalError> {
        let mut evaluator = Evaluator::new(X.view());
        evaluator.evaluate(&self.root)
    }

    /// Get string representation with format strings
    pub fn to_string_with_format(&self, format_strings: &[String]) -> String {
        self.render_node_with_format(&self.root, format_strings)
    }

    fn render_node_with_format(&self, node: &Node, formats: &[String]) -> String {
        match node {
            Node::Feature { name, .. } => name.clone(),
            Node::Constant(val) => format!("{:.3}", val),
            Node::Function { op_id, children, .. } => {
                let child_strs: Vec<String> = children
                    .iter()
                    .map(|c| self.render_node_with_format(c, formats))
                    .collect();

                // For now, use simple format - proper format string parsing can be added later
                if (*op_id as usize) < formats.len() {
                    // Simple placeholder replacement for common formats
                    let format_str = &formats[*op_id as usize];
                    let mut result = format_str.clone();
                    for (i, child_str) in child_strs.iter().enumerate() {
                        let placeholder = format!("{{{}}}", i);
                        result = result.replace(&placeholder, child_str);
                    }
                    result
                } else {
                    format!("op_{}({})", op_id, child_strs.join(", "))
                }
            }
        }
    }
}
