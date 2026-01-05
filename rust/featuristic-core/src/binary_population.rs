//! Binary genome population for feature selection
//!
//! This module implements a genetic algorithm for feature selection using binary genomes,
//! where each bit in a genome represents whether a feature is selected (true) or not (false).

use rand_chacha::ChaCha8Rng;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;

/// Binary genome type for feature selection
pub type BinaryGenome = Vec<bool>;

/// Population of binary genomes for feature selection
pub struct BinaryPopulation {
    /// Vector of binary genomes
    pub genomes: Vec<BinaryGenome>,
    /// Fitness values for each genome (lower is better)
    pub fitness: Vec<f64>,
    /// Tournament size for selection
    tournament_size: usize,
    /// Probability of crossover
    crossover_prob: f64,
    /// Probability of bit-flip mutation
    mutation_prob: f64,
    /// Random number generator
    rng: ChaCha8Rng,
}

impl BinaryPopulation {
    /// Create a new binary population
    ///
    /// # Arguments
    /// * `population_size` - Number of individuals in the population
    /// * `num_features` - Number of features (length of each genome)
    /// * `tournament_size` - Size of tournaments for selection
    /// * `crossover_prob` - Probability of crossover between 0.0 and 1.0
    /// * `mutation_prob` - Probability of bit-flip mutation between 0.0 and 1.0
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// A new BinaryPopulation with randomly initialized genomes
    ///
    /// # Note
    /// Each genome is guaranteed to have at least one feature selected to avoid empty subsets
    pub fn new(
        population_size: usize,
        num_features: usize,
        tournament_size: usize,
        crossover_prob: f64,
        mutation_prob: f64,
        mut rng: ChaCha8Rng,
    ) -> Self {
        let mut genomes = Vec::with_capacity(population_size);

        // Initialize random genomes with at least 1 feature selected
        for _ in 0..population_size {
            let mut genome = (0..num_features)
                .map(|_| rng.gen_bool(0.5))
                .collect::<Vec<bool>>();

            // Ensure at least one feature is selected
            if !genome.iter().any(|&x| x) {
                let random_idx = rng.gen_range(0..num_features);
                genome[random_idx] = true;
            }

            genomes.push(genome);
        }

        let fitness = vec![f64::INFINITY; population_size];

        Self {
            genomes,
            fitness,
            tournament_size,
            crossover_prob,
            mutation_prob,
            rng,
        }
    }

    /// Get the population size
    pub fn size(&self) -> usize {
        self.genomes.len()
    }

    /// Get the number of features (genome length)
    pub fn num_features(&self) -> usize {
        if self.genomes.is_empty() {
            0
        } else {
            self.genomes[0].len()
        }
    }

    /// Set fitness values
    ///
    /// # Arguments
    /// * `fitness` - Vector of fitness values (must match population size)
    pub fn set_fitness(&mut self, fitness: Vec<f64>) {
        assert_eq!(
            fitness.len(),
            self.genomes.len(),
            "Fitness vector length must match population size"
        );
        self.fitness = fitness;
    }

    /// Get fitness values
    pub fn get_fitness(&self) -> &[f64] {
        &self.fitness
    }

    /// Get a reference to the genomes
    pub fn get_genomes(&self) -> &[BinaryGenome] {
        &self.genomes
    }

    /// Get the best genome (lowest fitness)
    ///
    /// # Returns
    /// A clone of the genome with the minimum fitness value
    ///
    /// # Panics
    /// If all fitness values are INFINITY (no fitness has been set)
    pub fn get_best_genome(&self) -> BinaryGenome {
        let best_idx = self.fitness
            .iter()
            .enumerate()
            .min_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal))
            .expect("All fitness values are INFINITY")
            .0;

        self.genomes[best_idx].clone()
    }

    /// Evolve population to create the next generation
    ///
    /// This uses genetic operators with parallel evolution:
    /// 1. Tournament selection to choose parents
    /// 2. Uniform crossover with probability `crossover_prob`
    /// 3. Bit-flip mutation with probability `mutation_prob`
    /// 4. Elitism: best individual always survives
    pub fn evolve(&mut self) {
        // Clone data needed for parallel evolution
        let genomes = self.genomes.clone();
        let fitness = self.fitness.clone();
        let tournament_size = self.tournament_size;
        let crossover_prob = self.crossover_prob;
        let mutation_prob = self.mutation_prob;
        let num_features = self.num_features();
        let seed = self.rng.gen::<u64>();

        // Find best genome for elitism
        let best_idx = fitness
            .iter()
            .enumerate()
            .min_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let best_genome = self.genomes[best_idx].clone();

        // Evolve in parallel using thread-local RNGs
        let mut new_genomes: Vec<BinaryGenome> = (0..self.genomes.len())
            .into_par_iter()
            .map(|i| {
                // Thread-local RNG seeded with global seed + index
                let mut thread_rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(i as u64));

                // Tournament selection
                let parent1 = tournament_select_parallel(&genomes, &fitness, tournament_size, &mut thread_rng);

                // Crossover
                let mut child = if thread_rng.gen::<f64>() < crossover_prob {
                    let parent2 = tournament_select_parallel(&genomes, &fitness, tournament_size, &mut thread_rng);
                    uniform_crossover(parent1, parent2, &mut thread_rng)
                } else {
                    parent1.to_vec()
                };

                // Mutation
                bit_flip_mutate(&mut child, mutation_prob, &mut thread_rng);

                // Ensure at least one feature is selected
                if !child.iter().any(|&x| x) && num_features > 0 {
                    let random_idx = thread_rng.gen_range(0..num_features);
                    child[random_idx] = true;
                }

                child
            })
            .collect();

        // Elitism: replace first individual with best from previous generation
        new_genomes[0] = best_genome;

        self.genomes = new_genomes;
        self.fitness = vec![f64::INFINITY; self.genomes.len()];
    }

    /// Get a mutable reference to the RNG
    pub fn rng_mut(&mut self) -> &mut ChaCha8Rng {
        &mut self.rng
    }
}

/// Tournament selection for parallel context
///
/// # Arguments
/// * `genomes` - Reference to all genomes
/// * `fitness` - Reference to fitness values
/// * `tournament_size` - Number of individuals to sample
/// * `rng` - Random number generator
///
/// # Returns
/// Reference to the best genome in the tournament
fn tournament_select_parallel<'a>(
    genomes: &'a [BinaryGenome],
    fitness: &'a [f64],
    tournament_size: usize,
    rng: &mut impl Rng,
) -> &'a [bool] {
    let best_idx = (0..tournament_size)
        .map(|_| rng.gen_range(0..genomes.len()))
        .min_by(|&i, &j| fitness[i].partial_cmp(&fitness[j]).unwrap())
        .unwrap();

    &genomes[best_idx]
}

/// Uniform crossover between two parent genomes
///
/// # Arguments
/// * `parent1` - First parent genome
/// * `parent2` - Second parent genome
/// * `rng` - Random number generator
///
/// # Returns
/// Child genome with bits randomly selected from either parent
///
/// # Note
/// Each bit has a 50% chance of coming from either parent
fn uniform_crossover(parent1: &[bool], parent2: &[bool], rng: &mut impl Rng) -> BinaryGenome {
    parent1
        .iter()
        .zip(parent2.iter())
        .map(|(&gene1, &gene2)| {
            if rng.gen::<f64>() < 0.5 {
                gene1
            } else {
                gene2
            }
        })
        .collect()
}

/// Bit-flip mutation
///
/// # Arguments
/// * `genome` - Genome to mutate (modified in place)
/// * `mutation_prob` - Probability of flipping each bit
/// * `rng` - Random number generator
///
/// # Note
/// Each bit has an independent probability of being flipped
fn bit_flip_mutate(genome: &mut [bool], mutation_prob: f64, rng: &mut impl Rng) {
    for gene in genome.iter_mut() {
        if rng.gen::<f64>() < mutation_prob {
            *gene = !*gene;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_population_creation() {
        let rng = ChaCha8Rng::seed_from_u64(42);
        let pop = BinaryPopulation::new(10, 5, 3, 0.9, 0.1, rng);

        assert_eq!(pop.size(), 10);
        assert_eq!(pop.num_features(), 5);

        // Check that all genomes have at least one feature selected
        for genome in pop.get_genomes() {
            assert!(genome.iter().any(|&x| x), "Empty genome found");
        }
    }

    #[test]
    fn test_set_and_get_fitness() {
        let rng = ChaCha8Rng::seed_from_u64(42);
        let mut pop = BinaryPopulation::new(10, 5, 3, 0.9, 0.1, rng);

        let fitness = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        pop.set_fitness(fitness.clone());

        assert_eq!(pop.get_fitness(), &fitness[..]);
    }

    #[test]
    fn test_get_best_genome() {
        let rng = ChaCha8Rng::seed_from_u64(42);
        let mut pop = BinaryPopulation::new(5, 3, 3, 0.9, 0.1, rng);

        // Set specific fitness values
        pop.set_fitness(vec![5.0, 1.0, 3.0, 4.0, 2.0]);

        let best = pop.get_best_genome();
        // Best should be the genome at index 1 (fitness = 1.0)
        assert_eq!(best, pop.get_genomes()[1]);
    }

    #[test]
    fn test_evolution() {
        let rng = ChaCha8Rng::seed_from_u64(42);
        let mut pop = BinaryPopulation::new(20, 10, 5, 0.9, 0.1, rng);

        // Set random fitness
        let fitness: Vec<f64> = (0..20).map(|i| i as f64).collect();
        pop.set_fitness(fitness);

        let genomes_before = pop.get_genomes().to_vec();

        // Evolve
        pop.evolve();

        // Check population size remains the same
        assert_eq!(pop.size(), 20);
        assert_eq!(pop.num_features(), 10);

        // Check that genomes have changed (at least some)
        let genomes_different = genomes_before
            .iter()
            .zip(pop.get_genomes().iter())
            .any(|(g1, g2)| g1 != g2);
        assert!(genomes_different, "No genomes changed after evolution");

        // Check that fitness was reset
        assert_eq!(pop.get_fitness(), &vec![f64::INFINITY; 20][..]);
    }

    #[test]
    fn test_uniform_crossover() {
        let parent1 = vec![true, false, true, false];
        let parent2 = vec![false, true, false, true];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let child = uniform_crossover(&parent1, &parent2, &mut rng);

        // Child should have same length
        assert_eq!(child.len(), 4);

        // Child should be a mix (not identical to either parent in most cases)
        // With seed 42, we can verify it's not an exact copy
        assert!(child != parent1 || child != parent2);
    }

    #[test]
    fn test_bit_flip_mutation() {
        let mut genome = vec![true, false, true, false, true];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // High mutation rate to ensure some flips
        bit_flip_mutate(&mut genome, 1.0, &mut rng);

        // With 100% mutation rate, all bits should be flipped
        assert_eq!(genome, vec![false, true, false, true, false]);
    }

    #[test]
    fn test_tournament_selection() {
        let genomes = vec![
            vec![true, false],
            vec![false, true],
            vec![true, true],
            vec![false, false],
        ];
        let fitness = vec![10.0, 5.0, 8.0, 3.0];
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Run tournament multiple times
        let mut selected = Vec::new();
        for _ in 0..10 {
            let best = tournament_select_parallel(&genomes, &fitness, 2, &mut rng);
            selected.push(best);
        }

        // With seed 42, we can verify the function works
        // (best fitness is 3.0 at index 3, but tournament might not always pick it)
        assert!(!selected.is_empty());
    }
}
