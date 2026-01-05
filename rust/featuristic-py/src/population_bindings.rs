//! PyO3 bindings for population operations

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::{PyArray2, PyArray1, ToPyArray};
use featuristic_core::{SymbolicPopulation, SymbolicTree, Node};
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;

/// Parse a Python dict into a Rust Node (reused from tree_bindings)
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
            return Ok(Node::Feature {
                name: feature_name,
                index: 0, // Simplified: always use 0 for now
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

/// Convert Rust Node to Python dict
fn node_to_py_dict(node: &Node, py: Python) -> PyResult<PyObject> {
    let dict = PyDict::new(py);

    match node {
        Node::Constant(value) => {
            dict.set_item("value", *value)?;
        }
        Node::Feature { name, index } => {
            dict.set_item("feature_name", name)?;
            dict.set_item("index", *index)?;
        }
        Node::Function { op_id, arity, children } => {
            dict.set_item("op_id", *op_id)?;
            dict.set_item("arity", *arity)?;

            let py_children = PyList::empty(py);
            for child in children.iter() {
                py_children.append(node_to_py_dict(child, py)?)?;
            }
            dict.set_item("children", py_children)?;
        }
    }

    Ok(dict.into())
}

/// Population wrapper for Python
#[pyclass]
pub struct Population {
    inner: SymbolicPopulation,
}

#[pymethods]
impl Population {
    #[new]
    fn new(
        population_size: usize,
        feature_names: Vec<String>,
        _operations: Vec<PyObject>,
        tournament_size: usize,
        crossover_prob: f64,
        mutation_prob: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        use featuristic_core::builtins::default_builtins;

        let base_seed = seed.unwrap_or(42);
        let mut rng = ChaCha8Rng::seed_from_u64(base_seed);
        let operations = default_builtins();

        // Create initial population
        let mut population = SymbolicPopulation::new(
            population_size,
            tournament_size,
            crossover_prob,
            mutation_prob,
            rng,
        );

        // Generate random trees with balanced parameters
        // Use a seeded RNG for tree generation (different seed for each tree for variety)
        let trees: Vec<SymbolicTree> = (0..population_size)
            .enumerate()
            .map(|(i, _)| {
                let mut tree_rng = ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(i as u64));
                SymbolicTree::random(
                    5, // max_depth (shallow enough to prevent overflow, deep enough for interactions)
                    &feature_names,
                    &operations,
                    &mut tree_rng,
                    -10.0, 10.0, true, 0.15, 0.5,  // stop_prob: 50% chance of stopping (balanced)
                )
            })
            .collect();

        population.set_trees(trees);

        Ok(Self { inner: population })
    }

    /// Evaluate all trees in parallel
    fn evaluate_parallel(&mut self, x: &PyArray2<f64>) -> PyResult<Vec<PyObject>> {
        let x_view = unsafe { x.as_array() };

        let results = self.inner.evaluate_parallel(&x_view);

        Python::with_gil(|py| {
            let py_results: Result<Vec<_>, _> = results
                .into_iter()
                .map(|result| {
                    match result {
                        Ok(arr) => Ok(arr.to_pyarray(py).into()),
                        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("Evaluation error: {}", e)
                        )),
                    }
                })
                .collect();

            py_results
        })
    }

    /// Evolve population using genetic operators
    fn evolve(&mut self) -> PyResult<()> {
        self.inner.evolve();
        Ok(())
    }

    /// Set fitness values
    fn set_fitness(&mut self, fitness: Vec<f64>) -> PyResult<()> {
        self.inner.set_fitness(fitness);
        Ok(())
    }

    /// Get fitness values
    fn get_fitness(&self) -> PyResult<Vec<f64>> {
        Ok(self.inner.get_fitness().to_vec())
    }

    /// Get population size
    fn size(&self) -> usize {
        self.inner.size()
    }

    /// Get all trees as Python dicts
    fn get_trees(&self, py: Python) -> PyResult<Vec<PyObject>> {
        let trees = self.inner.get_trees();
        let mut py_trees = Vec::new();

        for tree in trees.iter() {
            let tree_dict = node_to_py_dict(&tree.root, py)?;
            py_trees.push(tree_dict);
        }

        Ok(py_trees)
    }

    /// Set trees from Python dicts
    fn set_trees(&mut self, tree_dicts: Vec<PyObject>) -> PyResult<()> {
        Python::with_gil(|py| {
            let mut trees = Vec::new();

            for tree_dict_obj in tree_dicts {
                let tree_dict: &PyDict = tree_dict_obj.as_ref(py).downcast()?;
                let node = parse_node_from_py(tree_dict)?;
                let tree = SymbolicTree::new(node);
                trees.push(tree);
            }

            self.inner.set_trees(trees);
            Ok(())
        })
    }

    /// Evaluate fitness for all programs given a Python callable fitness function
    ///
    /// This method evaluates all programs and computes fitness using the provided
    /// Python callable, avoiding the need for manual loops.
    #[pyo3(signature = (x, y, fitness_func, parsimony_coefficient=0.0))]
    fn evaluate_fitness(
        &mut self,
        x: &PyArray2<f64>,
        y: &PyArray1<f64>,
        fitness_func: &PyAny,
        parsimony_coefficient: f64,
    ) -> PyResult<PyObject> {
        // Get predictions using existing Rust method
        let predictions = self.evaluate_parallel(x)?;

        Python::with_gil(|py| {
            let mut fitness_scores = Vec::new();

            for pred_obj in predictions {
                let pred: &PyArray1<f64> = pred_obj.extract(py)?;

                // Call Python fitness function with y_true, y_pred
                let result = fitness_func.call1((y, pred))?;
                let score: f64 = result.extract()?;

                // Add parsimony penalty (simplified - could use tree depth)
                let penalty = parsimony_coefficient;
                fitness_scores.push(score + penalty);
            }

            // Return as Python list (simpler than numpy array conversion)
            Ok(fitness_scores.into_py(py))
        })
    }

    /// Get the k best programs with metadata
    ///
    /// Returns a list of dicts containing tree, fitness, and index for the
    /// k best programs (lowest fitness scores).
    #[pyo3(signature = (k=1))]
    fn get_best(&self, k: usize, py: Python) -> PyResult<Vec<PyObject>> {
        let fitness = self.inner.get_fitness();
        let trees = self.inner.get_trees();

        // Create indexed list and sort by fitness
        let mut indexed: Vec<(usize, f64)> = fitness
            .iter()
            .enumerate()
            .map(|(i, &f)| (i, f))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Get top k
        let results: Result<Vec<_>, _> = indexed
            .iter()
            .take(k)
            .map(|(idx, score)| {
                let tree = &trees[*idx];
                let tree_dict = node_to_py_dict(&tree.root, py)?;

                let result_dict = PyDict::new(py);
                result_dict.set_item("tree", tree_dict)?;
                result_dict.set_item("fitness", *score)?;
                result_dict.set_item("index", *idx)?;

                Ok(result_dict.into())
            })
            .collect();

        results
    }

    /// Evolve population for multiple generations with automatic tracking
    ///
    /// This is a convenience method that combines evaluation, fitness setting,
    /// and evolution into a single call with early stopping support.
    #[pyo3(signature = (
        x,
        y,
        fitness_func,
        n_generations,
        parsimony_coefficient=0.001,
        early_stopping=true,
        early_stopping_patience=5,
        show_progress_bar=false
    ))]
    fn evolve_generations(
        &mut self,
        x: &PyArray2<f64>,
        y: &PyArray1<f64>,
        fitness_func: &PyAny,
        n_generations: usize,
        parsimony_coefficient: f64,
        early_stopping: bool,
        early_stopping_patience: usize,
        show_progress_bar: bool,
    ) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let mut best_fitness = f64::INFINITY;
            let mut patience_counter = 0;
            let mut best_generation_idx = 0;
            let mut fitness_history = Vec::new();

            // Simple for loop for now (tqdm integration can be added later)
            for generation in 0..n_generations {
                // Evaluate fitness (returns Python list)
                let fitness_obj = self.evaluate_fitness(x, y, fitness_func, parsimony_coefficient)?;
                let fitness_vec: Vec<f64> = fitness_obj.extract(py)?;

                // Set fitness
                self.inner.set_fitness(fitness_vec.clone());

                // Check for improvement
                let current_best = fitness_vec
                    .iter()
                    .fold(f64::INFINITY, |a, &b| a.min(b));

                // Track fitness history
                fitness_history.push(current_best);

                if current_best < best_fitness {
                    best_fitness = current_best;
                    best_generation_idx = generation;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                }

                // Early stopping check
                if early_stopping && patience_counter >= early_stopping_patience {
                    break;
                }

                // Evolve (but not after last generation)
                if generation < n_generations - 1 {
                    self.inner.evolve();
                }
            }

            // Build result dict
            let result = PyDict::new(py);
            result.set_item("best_generation", best_generation_idx)?;
            result.set_item("best_fitness", best_fitness)?;
            result.set_item("stopped_early", patience_counter >= early_stopping_patience)?;
            result.set_item("fitness_history", fitness_history)?;

            Ok(result.into())
        })
    }
}
