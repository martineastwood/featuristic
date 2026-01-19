# Binary Genetic Algorithm for Feature Selection in Nim
#
# This implements a classic binary GA where each individual is a bitmask
# indicating which features to select.

import std/random
import std/algorithm
import std/sequtils
import std/math


# ============================================================================
# Types
# ============================================================================

type
  BinaryGenome* = seq[int]  # Sequence of 0s and 1s
  BinaryPopulation* = seq[BinaryGenome]


# ============================================================================
# Population Initialization
# ============================================================================

proc initBinaryPopulation*(
  populationSize: int,
  genomeLength: int,
  rng: var Rand
): BinaryPopulation =
  ## Initialize a random binary population

  result = newSeq[BinaryGenome](populationSize)

  for i in 0..<populationSize:
    var genome = newSeq[int](genomeLength)
    for j in 0..<genomeLength:
      genome[j] = rng.rand(1)  # Random 0 or 1
    result[i] = genome


# ============================================================================
# Fitness Evaluation
# ============================================================================

proc countSelected*(genome: BinaryGenome): int =
  ## Count how many features are selected (number of 1s)
  var count = 0
  for val in genome:
    if val == 1:
      inc(count)
  return count


# ============================================================================
# Selection
# ============================================================================

proc tournamentSelect*(
  population: BinaryPopulation,
  fitness: seq[float64],
  tournamentSize: int,
  rng: var Rand
): BinaryGenome =
  ## Select an individual using tournament selection

  let popSize = len(population)
  if popSize == 0:
    return newSeq[int](0)

  var bestIdx = rng.rand(popSize - 1)
  var bestFitness = fitness[bestIdx]

  for _ in 1..<tournamentSize:
    let idx = rng.rand(popSize - 1)
    if fitness[idx] < bestFitness:  # Lower is better
      bestFitness = fitness[idx]
      bestIdx = idx

  return population[bestIdx]


# ============================================================================
# Crossover (Single-Point)
# ============================================================================

proc singlePointCrossover*(
  parent1: BinaryGenome,
  parent2: BinaryGenome,
  crossoverProb: float64,
  rng: var Rand
): tuple[child1, child2: BinaryGenome] =
  ## Perform single-point crossover

  let genomeLength = len(parent1)

  # Default: no crossover, just copy parents
  result.child1 = parent1
  result.child2 = parent2

  if rng.rand(1.0) >= crossoverProb:
    return

  # Single-point crossover
  let point = rng.rand(genomeLength - 2) + 1  # Not at edges

  # Create children
  var child1 = newSeq[int](genomeLength)
  var child2 = newSeq[int](genomeLength)

  # Child 1: parent1[0:point] + parent2[point:]
  for i in 0..<point:
    child1[i] = parent1[i]
    child2[i] = parent2[i]

  for i in point..<genomeLength:
    child1[i] = parent2[i]
    child2[i] = parent1[i]

  result.child1 = child1
  result.child2 = child2


# ============================================================================
# Mutation (Bit Flip)
# ============================================================================

proc bitFlipMutate*(
  genome: BinaryGenome,
  mutationProb: float64,
  rng: var Rand
): BinaryGenome =
  ## Mutate a genome by flipping bits

  let genomeLength = len(genome)
  result = newSeq[int](genomeLength)

  for i in 0..<genomeLength:
    if rng.rand(1.0) < mutationProb:
      # Flip bit: 0 -> 1, 1 -> 0
      result[i] = 1 - genome[i]
    else:
      result[i] = genome[i]


# ============================================================================
# Evolution
# ============================================================================

proc evolveBinaryPopulation*(
  population: BinaryPopulation,
  fitness: seq[float64],
  crossoverProb: float64,
  mutationProb: float64,
  tournamentSize: int,
  rng: var Rand
): BinaryPopulation =
  ## Evolve the binary population by one generation

  let popSize = len(population)

  # Selection and reproduction
  var newPopulation = newSeq[BinaryGenome](popSize)

  var i = 0
  while i < popSize:
    # Select two parents
    let parent1 = tournamentSelect(population, fitness, tournamentSize, rng)
    let parent2 = tournamentSelect(population, fitness, tournamentSize, rng)

    # Crossover
    let (child1, child2) = singlePointCrossover(parent1, parent2, crossoverProb, rng)

    # Mutate
    let mutatedChild1 = bitFlipMutate(child1, mutationProb, rng)
    var mutatedChild2 = bitFlipMutate(child2, mutationProb, rng)

    # Add to new population
    newPopulation[i] = mutatedChild1
    inc(i)

    if i < popSize:
      newPopulation[i] = mutatedChild2
      inc(i)

  return newPopulation


# ============================================================================
# Export for Python
# ============================================================================

export initBinaryPopulation,
       tournamentSelect,
       singlePointCrossover,
       bitFlipMutate,
       evolveBinaryPopulation,
       countSelected
