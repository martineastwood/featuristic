//! Featuristic Core Library
//!
//! This library contains the core genetic programming and symbolic regression
//! algorithms implemented in Rust for maximum performance.

pub mod tree;
pub mod evaluate;
pub mod builtins;
pub mod population;
pub mod genetic;
pub mod mrmr;
pub mod rng;
pub mod binary_population;

// Re-exports for convenience
pub use tree::{Node, SymbolicTree, SymbolicOp};
pub use population::SymbolicPopulation;
pub use binary_population::{BinaryPopulation, BinaryGenome};
