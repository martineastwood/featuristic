//! Random number generation utilities

use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;

/// Create a seeded ChaCha8 RNG for reproducibility
pub fn seeded_rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}
