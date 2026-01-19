# Genetic operations for symbolic regression in Nim
# Provides crossover, mutation, selection for stack-based programs

import std/random
import std/tables
import std/algorithm
import ../core/types
import ../core/program


# ============================================================================
# Program Utilities
# ============================================================================

proc countNodes*(program: StackProgram): int =
  ## Count the number of nodes in a program
  return len(program.nodes)


proc getRandomNodeIndex*(program: StackProgram, rng: var Rand): int =
  ## Select a random node index from the program
  ## Uses weighted probability based on depth
  let numNodes = len(program.nodes)
  if numNodes == 0:
    return -1

  # Simple uniform selection for now
  ## TODO: Use depth-weighted selection like Python implementation
  return rng.rand(numNodes - 1)


# ============================================================================
# Tree Manipulation (on flat post-order representation)
# ============================================================================

proc cloneSubtree*(program: StackProgram, nodeIdx: int): tuple[nodes: seq[StackProgramNode], mapping: Table[int, int]] =
  ## Clone a subtree rooted at nodeIdx

  ## Returns:
  ##   - nodes: The cloned subtree nodes
  ##   - mapping: Map from original node indices to cloned node indices

  if nodeIdx < 0 or nodeIdx >= len(program.nodes):
    return (newSeq[StackProgramNode](0), initTable[int, int]())

  var result = newSeq[StackProgramNode](0)
  var mapping = initTable[int, int]()

  # For post-order representation, a subtree consists of:
  # - The root node and all its descendants
  # - All descendants have indices < root index in post-order

  # Find all nodes in the subtree
  var subtreeIndices = newSeq[int](0)

  proc collectDescendants(idx: int) =
    ## Collect all descendant indices
    if idx < 0:
      return

    subtreeIndices.add(idx)

    let node = program.nodes[idx]
    if node.left >= 0 and node.left < idx:
      collectDescendants(node.left)
    if node.right >= 0 and node.right < idx:
      collectDescendants(node.right)

  collectDescendants(nodeIdx)

  # Sort indices to maintain post-order
  subtreeIndices.sort(cmp[int])

  # Clone nodes and create mapping
  var newIndex = 0
  for oldIdx in subtreeIndices:
    mapping[oldIdx] = newIndex
    result.add(program.nodes[oldIdx])
    inc(newIndex)

  # Update child indices in cloned nodes
  for i in 0..<len(result):
    var node = result[i]
    if node.left >= 0:
      if node.left in mapping:
        node.left = mapping[node.left]
      else:
        node.left = -1
    if node.right >= 0:
      if node.right in mapping:
        node.right = mapping[node.right]
      else:
        node.right = -1
    result[i] = node

  return (result, mapping)


proc replaceSubtree*(program: var StackProgram, targetIdx: int, replacementNodes: seq[StackProgramNode], rootOffset: int) =
  ## Replace a subtree rooted at targetIdx with new nodes

  ## Args:
  ##   - program: The program to modify
  ##   - targetIdx: Index of the root of the subtree to replace
  ##   - replacementNodes: The new subtree nodes (already in post-order)
  ##   - rootOffset: Offset to add to child indices when replacing

  ## This is complex because:
  ## 1. We need to remove the old subtree
  ## 2. Insert the new subtree
  ## 3. Update all parent references

  # For simplicity, we'll implement this as:
  # 1. Create new program array
  # 2. Copy nodes before the subtree
  # 3. Insert new nodes
  # 4. Copy nodes after the subtree (with adjusted indices)

  # Find the size of the subtree to replace
  var subtreeSize = 0

  proc countSubtree(program: StackProgram, idx: int): int =
    ## Count nodes in subtree
    if idx < 0 or idx >= len(program.nodes):
      return 0
    var count = 1
    let node = program.nodes[idx]
    if node.left >= 0 and node.left < idx:
      count += countSubtree(program, node.left)
    if node.right >= 0 and node.right < idx:
      count += countSubtree(program, node.right)
    return count

  subtreeSize = countSubtree(program, targetIdx)

  # Create new node array
  var newNodes = newSeq[StackProgramNode](0)

  # Copy nodes before the subtree (indices 0 to targetIdx - subtreeSize)
  let beforeCount = targetIdx - subtreeSize
  for i in 0..<beforeCount:
    newNodes.add(program.nodes[i])

  # Insert new subtree nodes
  let newRootIdx = len(newNodes)
  for i in 0..<len(replacementNodes):
    # Adjust child indices based on new position
    var node = replacementNodes[i]
    if node.left >= 0:
      node.left = node.left + newRootIdx
    if node.right >= 0:
      node.right = node.right + newRootIdx
    newNodes.add(node)

  # Copy nodes after the subtree (indices targetIdx+1 to end)
  # These need their indices adjusted
  let afterStart = targetIdx + 1
  for i in afterStart..<len(program.nodes):
    var node = program.nodes[i]

    # Check if this node or its ancestors reference the replaced subtree
    # This is complex - for now, just copy without adjustment
    # TODO: Implement proper index adjustment
    newNodes.add(node)

  program.nodes = newNodes


# ============================================================================
# Genetic Operations
# ============================================================================

proc crossover*(parent1: StackProgram, parent2: StackProgram, rng: var Rand): StackProgram =
  ## Perform crossover between two programs
  ## Simplified strategy: randomly select either parent1 or parent2
  ## This introduces diversity while maintaining solution quality

  # 70% chance to return parent1 (maintain good solutions)
  # 30% chance to return parent2 (introduce diversity)
  if rng.rand(1.0) < 0.7:
    return StackProgram(nodes: parent1.nodes, depth: parent1.depth)
  else:
    return StackProgram(nodes: parent2.nodes, depth: parent2.depth)


proc mutate*(program: StackProgram, newXProgram: StackProgram, rng: var Rand): StackProgram =
  ## Mutate a program by replacing it with a new random program
  ## Simplified strategy: randomly return original or new program

  # 80% chance to keep original program
  # 20% chance to replace with new random program
  if rng.rand(1.0) < 0.8:
    return StackProgram(nodes: program.nodes, depth: program.depth)
  else:
    return StackProgram(nodes: newXProgram.nodes, depth: newXProgram.depth)


proc tournamentSelect*(population: seq[StackProgram], fitness: seq[float64],
                       tournamentSize: int, rng: var Rand): StackProgram =
  ## Select a program using tournament selection

  ## Args:
  ##   - population: The population of programs
  ##   - fitness: Fitness values (lower is better)
  ##   - tournamentSize: Number of individuals in each tournament
  ##   - rng: Random number generator

  ## Returns: The selected program

  let popSize = len(population)
  if popSize == 0:
    return nil

  var bestIdx = -1
  var bestFitness = Inf

  # Run tournament
  for _ in 0..<tournamentSize:
    let idx = rng.rand(popSize - 1)
    if fitness[idx] < bestFitness:
      bestFitness = fitness[idx]
      bestIdx = idx

  return population[bestIdx]


# Export functions without nuwa_export (implementation only)
export countNodes, getRandomNodeIndex, cloneSubtree, replaceSubtree,
       crossover, mutate, tournamentSelect
