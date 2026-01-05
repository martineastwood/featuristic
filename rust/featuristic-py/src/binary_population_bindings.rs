//! PyO3 bindings for binary population operations (feature selection)

use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray2, PyArray1, ToPyArray};
use featuristic_core::{BinaryPopulation as CoreBinaryPopulation, BinaryGenome};
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use ndarray::Axis;

/// BinaryPopulation wrapper for Python
///
/// This class provides a genetic algorithm for feature selection using binary genomes,
/// where each bit represents whether a feature is selected (true) or not (false).
#[pyclass]
pub struct BinaryPopulation {
    inner: CoreBinaryPopulation,
}

#[pymethods]
impl BinaryPopulation {
    /// Create a new binary population for feature selection
    ///
    /// # Arguments
    /// * `population_size` - Number of individuals in the population
    /// * `num_features` - Number of features (length of each genome)
    /// * `tournament_size` - Size of tournaments for selection (default: 10)
    /// * `crossover_prob` - Probability of crossover between 0.0 and 1.0 (default: 0.9)
    /// * `mutation_prob` - Probability of bit-flip mutation between 0.0 and 1.0 (default: 0.1)
    /// * `seed` - Random seed for reproducibility (default: None)
    ///
    /// # Returns
    /// A new BinaryPopulation instance
    ///
    /// # Example
    /// ```python
    /// import featuristic
    /// pop = featuristic.BinaryPopulation(
    ///     population_size=50,
    ///     num_features=10,
    ///     tournament_size=5,
    ///     crossover_prob=0.8,
    ///     mutation_prob=0.05,
    ///     seed=42
    /// )
    /// ```
    #[new]
    #[pyo3(signature = (
        population_size,
        num_features,
        tournament_size=10,
        crossover_prob=0.9,
        mutation_prob=0.1,
        seed=None
    ))]
    fn new(
        population_size: usize,
        num_features: usize,
        tournament_size: usize,
        crossover_prob: f64,
        mutation_prob: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let inner = CoreBinaryPopulation::new(
            population_size,
            num_features,
            tournament_size,
            crossover_prob,
            mutation_prob,
            if let Some(s) = seed {
                ChaCha8Rng::seed_from_u64(s)
            } else {
                ChaCha8Rng::from_entropy()
            },
        );

        Ok(Self { inner })
    }

    /// Evaluate fitness using Python objective function
    ///
    /// # Arguments
    /// * `x` - Input features as 2D numpy array (n_samples, n_features)
    /// * `y` - Target values as 1D numpy array
    /// * `objective_func` - Python callable with signature (X_subset, y) -> float
    ///
    /// # Returns
    /// List of fitness values (one per genome)
    ///
    /// # Example
    /// ```python
    /// import numpy as np
    /// from sklearn.linear_model import Ridge
    /// from sklearn.metrics import mean_squared_error
    ///
    /// X = np.random.randn(100, 10)
    /// y = np.random.randn(100)
    ///
    /// def objective(X_subset, y):
    ///     model = Ridge().fit(X_subset, y)
    ///     return mean_squared_error(y, model.predict(X_subset))
    ///
    /// fitness = pop.evaluate_fitness(X, y, objective)
    /// pop.set_fitness(fitness)
    /// ```
    fn evaluate_fitness(
        &mut self,
        x: &PyArray2<f64>,
        y: &PyArray1<f64>,
        objective_func: &PyAny,
    ) -> PyResult<PyObject> {
        let x_view = unsafe { x.as_array() };
        let genomes = self.inner.get_genomes();

        Python::with_gil(|py| {
            let mut fitness_scores = Vec::new();

            for genome in genomes.iter() {
                // Convert binary genome to column indices
                let selected_indices: Vec<usize> = genome
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &selected)| if selected { Some(idx) } else { None })
                    .collect();

                // Skip if no features selected
                if selected_indices.is_empty() {
                    fitness_scores.push(f64::INFINITY);
                    continue;
                }

                // Select columns from X
                let x_subset = x_view.select(Axis(1), &selected_indices);

                // Convert to numpy array
                let x_subset_py = x_subset.to_pyarray(py);

                // Call Python objective function
                let result = objective_func.call1((x_subset_py, y))?;

                // Extract fitness score
                let score: f64 = result.extract()?;

                fitness_scores.push(score);
            }

            // Return as Python list
            Ok(fitness_scores.into_py(py))
        })
    }

    /// Evolve population for multiple generations with early stopping
    ///
    /// # Arguments
    /// * `x` - Input features as 2D numpy array (n_samples, n_features)
    /// * `y` - Target values as 1D numpy array
    /// * `objective_func` - Python callable with signature (X_subset, y) -> float
    /// * `n_generations` - Maximum number of generations to evolve
    /// * `early_stopping` - Whether to use early stopping (default: True)
    /// * `early_stopping_patience` - Generations to wait for improvement (default: 5)
    /// * `show_progress_bar` - Whether to show progress bar (default: False, not yet implemented)
    ///
    /// # Returns
    /// Dictionary with keys:
    /// - 'best_generation': Index of the generation with best fitness
    /// - 'best_fitness': Best fitness value found
    /// - 'stopped_early': Whether evolution stopped early due to convergence
    ///
    /// # Example
    /// ```python
    /// result = pop.evolve_generations(
    ///     X, y, objective_func,
    ///     n_generations=50,
    ///     early_stopping=True,
    ///     early_stopping_patience=10
    /// )
    /// print(f"Best fitness: {result['best_fitness']}")
    /// ```
    #[pyo3(signature = (
        x,
        y,
        objective_func,
        n_generations,
        early_stopping=true,
        early_stopping_patience=5,
        show_progress_bar=false
    ))]
    fn evolve_generations(
        &mut self,
        x: &PyArray2<f64>,
        y: &PyArray1<f64>,
        objective_func: &PyAny,
        n_generations: usize,
        early_stopping: bool,
        early_stopping_patience: usize,
        show_progress_bar: bool,
    ) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let mut best_fitness = f64::INFINITY;
            let mut patience_counter = 0;
            let mut best_generation_idx = 0;
            let mut fitness_history = Vec::new();

            // Simple for loop (progress bar can be added later with indicatif)
            for generation in 0..n_generations {
                // Evaluate fitness (returns Python list)
                let fitness_obj = self.evaluate_fitness(x, y, objective_func)?;
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

    /// Get the best genome (lowest fitness)
    ///
    /// # Returns
    /// List of booleans representing the best feature subset
    ///
    /// # Example
    /// ```python
    /// best_genome = pop.get_best_genome()
    /// selected_features = [i for i, selected in enumerate(best_genome) if selected]
    /// ```
    fn get_best_genome(&self) -> PyResult<Vec<bool>> {
        Ok(self.inner.get_best_genome())
    }

    /// Get population size
    ///
    /// # Returns
    /// Number of individuals in the population
    fn size(&self) -> usize {
        self.inner.size()
    }

    /// Get number of features
    ///
    /// # Returns
    /// Length of each genome (number of features)
    fn num_features(&self) -> usize {
        self.inner.num_features()
    }

    /// Get fitness values
    ///
    /// # Returns
    /// List of fitness values (one per individual)
    fn get_fitness(&self) -> PyResult<Vec<f64>> {
        Ok(self.inner.get_fitness().to_vec())
    }

    /// Set fitness values
    ///
    /// # Arguments
    /// * `fitness` - List of fitness values (must match population size)
    ///
    /// # Example
    /// ```python
    /// fitness = [1.0, 2.0, 3.0, ...]  # Same length as population
    /// pop.set_fitness(fitness)
    /// ```
    fn set_fitness(&mut self, fitness: Vec<f64>) -> PyResult<()> {
        self.inner.set_fitness(fitness);
        Ok(())
    }

    /// Get all genomes
    ///
    /// # Returns
    /// List of binary genomes (each genome is a list of booleans)
    ///
    /// # Example
    /// ```python
    /// genomes = pop.get_genomes()
    /// for i, genome in enumerate(genomes):
    ///     print(f"Individual {i}: {genome}")
    /// ```
    fn get_genomes(&self, py: Python) -> PyResult<Vec<PyObject>> {
        let genomes = self.inner.get_genomes();
        let py_genomes: Vec<_> = genomes
            .iter()
            .map(|genome| genome.clone().into_py(py))
            .collect();

        Ok(py_genomes)
    }

    /// Evolve population by one generation
    ///
    /// This advances the population to the next generation using:
    /// 1. Tournament selection
    /// 2. Uniform crossover
    /// 3. Bit-flip mutation
    /// 4. Elitism (best individual always survives)
    ///
    /// # Note
    /// Fitness values are reset to infinity after evolution
    ///
    /// # Example
    /// ```python
    /// # Evaluate and set fitness
    /// fitness = pop.evaluate_fitness(X, y, objective)
    /// pop.set_fitness(fitness)
    ///
    /// # Evolve one generation
    /// pop.evolve()
    ///
    /// # Need to re-evaluate fitness after evolution
    /// fitness = pop.evaluate_fitness(X, y, objective)
    /// pop.set_fitness(fitness)
    /// ```
    fn evolve(&mut self) -> PyResult<()> {
        self.inner.evolve();
        Ok(())
    }
}
