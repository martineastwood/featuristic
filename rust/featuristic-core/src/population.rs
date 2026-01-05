//! Population management for genetic programming

use crate::tree::{SymbolicTree, Node};
use rand_chacha::ChaCha8Rng;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;
use ndarray::ArrayView2;

/// Population of symbolic trees
pub struct SymbolicPopulation {
    pub trees: Vec<SymbolicTree>,
    pub fitness: Vec<f64>,
    tournament_size: usize,
    crossover_prob: f64,
    mutation_prob: f64,
    rng: ChaCha8Rng,
}

impl SymbolicPopulation {
    /// Create a new population
    pub fn new(
        population_size: usize,
        tournament_size: usize,
        crossover_prob: f64,
        mutation_prob: f64,
        rng: ChaCha8Rng,
    ) -> Self {
        Self {
            trees: Vec::with_capacity(population_size),
            fitness: Vec::with_capacity(population_size),
            tournament_size,
            crossover_prob,
            mutation_prob,
            rng,
        }
    }

    /// Create a population from existing trees
    pub fn from_trees(
        trees: Vec<SymbolicTree>,
        tournament_size: usize,
        crossover_prob: f64,
        mutation_prob: f64,
        rng: ChaCha8Rng,
    ) -> Self {
        let population_size = trees.len();
        Self {
            trees,
            fitness: vec![f64::INFINITY; population_size],
            tournament_size,
            crossover_prob,
            mutation_prob,
            rng,
        }
    }

    /// Get the population size
    pub fn size(&self) -> usize {
        self.trees.len()
    }

    /// Get a reference to the trees
    pub fn get_trees(&self) -> &[SymbolicTree] {
        &self.trees
    }

    /// Get a mutable reference to the RNG
    pub fn rng_mut(&mut self) -> &mut ChaCha8Rng {
        &mut self.rng
    }

    /// Set the trees in the population
    pub fn set_trees(&mut self, trees: Vec<SymbolicTree>) {
        self.trees = trees;
        self.fitness = vec![f64::INFINITY; self.trees.len()];
    }

    /// Set fitness values
    pub fn set_fitness(&mut self, fitness: Vec<f64>) {
        self.fitness = fitness;
    }

    /// Get fitness values
    pub fn get_fitness(&self) -> &[f64] {
        &self.fitness
    }

    /// Evaluate all trees in parallel using Rayon
    pub fn evaluate_parallel(&self, x: &ArrayView2<f64>) -> Vec<Result<ndarray::Array1<f64>, crate::evaluate::EvalError>> {
        self.trees
            .par_iter()  // Parallel iterator
            .map(|tree| tree.evaluate(x))
            .collect()
    }

    /// Evolve population using genetic operators
    pub fn evolve(&mut self) {
        // Clone data needed for parallel evolution
        let trees = self.trees.clone();
        let fitness = self.fitness.clone();
        let tournament_size = self.tournament_size;
        let crossover_prob = self.crossover_prob;
        let mutation_prob = self.mutation_prob;
        let seed = self.rng.gen::<u64>();

        // Evolve in parallel using thread-local RNGs
        let new_trees: Vec<SymbolicTree> = (0..self.trees.len())
            .into_par_iter()
            .map(|i| {
                // Thread-local RNG seeded with global seed + index
                let mut thread_rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(i as u64));

                let parent1 = tournament_select_parallel(&trees, &fitness, tournament_size, &mut thread_rng);

                if thread_rng.gen::<f64>() < crossover_prob {
                    let parent2 = tournament_select_parallel(&trees, &fitness, tournament_size, &mut thread_rng);
                    crossover_trees(parent1, parent2, &mut thread_rng)
                } else if thread_rng.gen::<f64>() < mutation_prob {
                    mutate_tree(parent1, &mut thread_rng)
                } else {
                    parent1.clone()
                }
            })
            .collect();

        self.trees = new_trees;
        self.fitness = vec![f64::INFINITY; self.trees.len()];
    }

    /// Tournament selection - select best individual from random tournament
    fn tournament_select(&mut self, fitness: &[f64]) -> &SymbolicTree {
        let best_idx = (0..self.tournament_size)
            .map(|_| self.rng.gen_range(0..self.trees.len()))
            .min_by(|&i, &j| fitness[i].partial_cmp(&fitness[j]).unwrap())
            .unwrap();

        &self.trees[best_idx]
    }
}

/// Standalone helper functions for parallel evolution

/// Tournament selection for parallel context
fn tournament_select_parallel<'a>(
    trees: &'a [SymbolicTree],
    fitness: &[f64],
    tournament_size: usize,
    rng: &mut impl Rng,
) -> &'a SymbolicTree {
    let best_idx = (0..tournament_size)
        .map(|_| rng.gen_range(0..trees.len()))
        .min_by(|&i, &j| fitness[i].partial_cmp(&fitness[j]).unwrap())
        .unwrap();

    &trees[best_idx]
}

/// Subtree crossover for parallel context
fn crossover_trees(
    parent1: &SymbolicTree,
    parent2: &SymbolicTree,
    rng: &mut impl Rng,
) -> SymbolicTree {
    // Clone parent1 as the base
    let mut child = parent1.clone();

    // Select a random crossover point in child
    let crossover_result = find_crossover_point_helper(&mut child.root, rng);

    if let Some((path, _depth)) = crossover_result {
        // Select a random subtree from parent2
        let subtree2 = select_random_subtree(&parent2.root, rng);

        // Replace the subtree at the crossover point
        replace_subtree_helper(&mut child.root, &path, &subtree2);

        // Recalculate depth
        child.depth = crate::tree::SymbolicTree::calculate_depth(&child.root);
    }

    child
}

/// Subtree mutation for parallel context
fn mutate_tree(
    parent: &SymbolicTree,
    rng: &mut impl Rng,
) -> SymbolicTree {
    let mut child = parent.clone();

    // Select a random mutation point
    if let Some((path, _)) = find_crossover_point_helper(&mut child.root, rng) {
        // Generate a new random subtree
        let max_depth = std::cmp::min(child.get_depth(), 5); // Limit mutation depth
        let feature_names = extract_feature_names_helper(&child.root);
        let operations = crate::builtins::default_builtins();

        let new_subtree = SymbolicTree::random(
            max_depth,
            &feature_names,
            &operations,
            rng,
            -10.0, 10.0, true, 0.15, 0.6,
        );

        // Replace the subtree
        replace_subtree_helper(&mut child.root, &path, &new_subtree.root);

        // Recalculate depth
        child.depth = crate::tree::SymbolicTree::calculate_depth(&child.root);
    }

    child
}

/// Find a random crossover point in the tree
fn find_crossover_point_helper<R: Rng>(
    node: &mut Node,
    rng: &mut R,
) -> Option<(Vec<usize>, usize)> {
    let mut path = Vec::new();
    let mut current_node = node;
    let mut depth = 0;

    loop {
        match current_node {
            Node::Feature { .. } | Node::Constant(_) => {
                // Leaf node - return this as crossover point
                return Some((path, depth));
            }
            Node::Function { children, .. } => {
                if children.is_empty() {
                    return Some((path, depth));
                }

                // Decide whether to stop at this function node or go deeper
                if rng.gen::<f64>() < 0.5 {
                    return Some((path, depth));
                }

                // Go deeper into a random child
                let child_idx = rng.gen_range(0..children.len());
                path.push(child_idx);
                depth += 1;

                // Get mutable reference to child (unsafe, but we control the tree structure)
                let child_ptr = &mut children[child_idx] as *mut Node;
                current_node = unsafe { &mut *child_ptr };
            }
        }
    }
}

/// Select a random subtree from a tree
fn select_random_subtree<R: Rng>(root: &Node, rng: &mut R) -> Node {
    let mut current = root;

    loop {
        match current {
            Node::Feature { .. } | Node::Constant(_) => {
                return current.clone();
            }
            Node::Function { children, .. } => {
                if children.is_empty() {
                    return current.clone();
                }

                // Bias towards deeper nodes (like Python implementation)
                if !children.is_empty() && rng.gen::<f64>() < 0.3 {
                    return current.clone();
                }

                let child_idx = rng.gen_range(0..children.len());
                current = &children[child_idx];
            }
        }
    }
}

/// Replace a subtree at the given path
fn replace_subtree_helper(root: &mut Node, path: &[usize], new_subtree: &Node) {
    if path.is_empty() {
        *root = new_subtree.clone();
        return;
    }

    let first = path[0];
    if let Node::Function { ref mut children, .. } = root {
        if first < children.len() {
            replace_subtree_helper(&mut children[first], &path[1..], new_subtree);
        }
    }
}

/// Extract feature names from a tree
fn extract_feature_names_helper(root: &Node) -> Vec<String> {
    let mut names = Vec::new();
    collect_feature_names_helper(root, &mut names);
    names.sort();
    names.dedup();
    names
}

/// Helper to collect feature names
fn collect_feature_names_helper(node: &Node, names: &mut Vec<String>) {
    match node {
        Node::Feature { name, .. } => {
            if !names.contains(name) {
                names.push(name.clone());
            }
        }
        Node::Function { children, .. } => {
            for child in children {
                collect_feature_names_helper(child, names);
            }
        }
        Node::Constant(_) => {}
    }
}
