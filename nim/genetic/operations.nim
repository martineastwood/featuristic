# Genetic operations for symbolic regression in Nim
# Provides crossover, mutation, selection for stack-based programs

import std/random
import std/tables
import std/math
import ../core/types


# ============================================================================
# Program Utilities
# ============================================================================

proc countNodes*(program: StackProgram): int =
  ## Count the number of nodes in a program
  return len(program.nodes)


proc markSubtreeNodes(
  program: StackProgram,
  idx: int,
  inSubtree: var seq[bool]
) =
  ## Helper to mark all nodes in a subtree
  if idx < 0 or idx >= len(program.nodes):
    return
  inSubtree[idx] = true
  let node = program.nodes[idx]
  if node.left >= 0 and node.left < idx:
    markSubtreeNodes(program, node.left, inSubtree)
  if node.right >= 0 and node.right < idx:
    markSubtreeNodes(program, node.right, inSubtree)


proc findSubtreeBounds*(program: StackProgram, rootIdx: int): tuple[startIdx: int, size: int] =
  ## Find the bounds (start index and size) of a subtree in post-order representation
  ##
  ## In post-order:
  ## - All descendants of a node have indices < node index
  ## - The subtree starts at the first descendant (lowest index)
  ## - The subtree includes the root at the highest index

  if rootIdx < 0 or rootIdx >= len(program.nodes):
    return (startIdx: -1, size: 0)

  # Find all nodes in the subtree
  var inSubtree = newSeq[bool](len(program.nodes))
  for i in 0..<len(inSubtree):
    inSubtree[i] = false

  markSubtreeNodes(program, rootIdx, inSubtree)

  # Find start (first marked node) and count
  var startIdx = -1
  var size = 0
  for i in 0..<len(inSubtree):
    if inSubtree[i]:
      if startIdx < 0:
        startIdx = i
      inc(size)

  return (startIdx: startIdx, size: size)


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

  var clonedNodes = newSeq[StackProgramNode](0)
  var mapping = initTable[int, int]()

  # For post-order representation, a subtree consists of:
  # - The root node and all its descendants
  # - All descendants have indices < root index in post-order

  # Find all nodes in the subtree
  let (startIdx, size) = findSubtreeBounds(program, nodeIdx)

  # Clone nodes and create mapping
  var newIndex = 0
  for i in countup(startIdx, startIdx + size - 1):
    mapping[i] = newIndex
    clonedNodes.add(program.nodes[i])
    inc(newIndex)

  # Update child indices in cloned nodes
  for i in 0..<len(clonedNodes):
    var node = clonedNodes[i]
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
    clonedNodes[i] = node

  return (clonedNodes, mapping)


proc replaceSubtree*(program: var StackProgram, targetIdx: int, replacementNodes: seq[StackProgramNode]) =
  ## Replace a subtree rooted at targetIdx with new nodes
  ##
  ## This properly handles index adjustments in post-order representation

  if targetIdx < 0 or targetIdx >= len(program.nodes):
    return

  if len(replacementNodes) == 0:
    return

  # Find the bounds of the subtree to replace
  let (startIdx, oldSize) = findSubtreeBounds(program, targetIdx)

  # Create new node array
  var newNodes = newSeq[StackProgramNode](0)

  # Copy nodes before the subtree
  for i in 0..<startIdx:
    newNodes.add(program.nodes[i])

  # Track the index of the new subtree root
  let newRootIdx = len(newNodes) + len(replacementNodes) - 1

  # Insert new subtree nodes with adjusted indices
  var newMapping = initTable[int, int]()
  for i in 0..<len(replacementNodes):
    let oldIdx = startIdx + i
    newMapping[oldIdx] = len(newNodes)

    var node = replacementNodes[i]
    # Adjust child indices to point within new subtree
    if node.left >= 0 and node.left < len(replacementNodes):
      node.left = newRootIdx - (len(replacementNodes) - 1 - node.left)
    if node.right >= 0 and node.right < len(replacementNodes):
      node.right = newRootIdx - (len(replacementNodes) - 1 - node.right)
    newNodes.add(node)

  # Copy nodes after the subtree, updating their child references
  let sizeDiff = len(replacementNodes) - oldSize
  for i in countup(startIdx + oldSize, len(program.nodes) - 1):
    var node = program.nodes[i]

    # Update child indices if they referenced nodes in the old subtree
    if node.left >= startIdx and node.left < startIdx + oldSize:
      # Pointed to old subtree, now point to new root
      node.left = newRootIdx
    elif node.left >= startIdx + oldSize:
      # Pointed to node after old subtree, adjust for size difference
      node.left = node.left + sizeDiff

    if node.right >= startIdx and node.right < startIdx + oldSize:
      # Pointed to old subtree, now point to new root
      node.right = newRootIdx
    elif node.right >= startIdx + oldSize:
      # Pointed to node after old subtree, adjust for size difference
      node.right = node.right + sizeDiff

    newNodes.add(node)

  program.nodes = newNodes


# ============================================================================
# Genetic Operations
# ============================================================================

proc crossover*(parent1: StackProgram, parent2: StackProgram, rng: var Rand, maxDepth: int = 6): StackProgram =
  ## Perform subtree crossover between two programs
  ##
  ## Strategy:
  ## 1. Select random subtree from parent1
  ## 2. Select random subtree from parent2
  ## 3. Replace parent1's subtree with parent2's subtree
  ## 4. Return offspring (clone of parent1 with replacement) IF depth limit not exceeded

  if len(parent1.nodes) == 0 or len(parent2.nodes) == 0:
    # One parent is empty, return the other
    if len(parent1.nodes) > 0:
      return StackProgram(nodes: parent1.nodes, depth: parent1.depth)
    else:
      return StackProgram(nodes: parent2.nodes, depth: parent2.depth)

  # Clone parent1 to modify
  var offspring = StackProgram(nodes: parent1.nodes, depth: parent1.depth)

  # Try crossover up to 3 times to find one that doesn't exceed depth limit
  for attempt in 0..<3:
    # Select random crossover point in offspring (from parent1)
    let targetIdx = getRandomNodeIndex(offspring, rng)
    if targetIdx < 0:
      break

    # Select random subtree from parent2 to insert
    let sourceIdx = getRandomNodeIndex(parent2, rng)
    if sourceIdx < 0:
      break

    # Clone subtree from parent2
    let (subtreeNodes, _) = cloneSubtree(parent2, sourceIdx)

    # Check if replacement would exceed max depth
    let currentDepth = len(offspring.nodes) div 2  # Rough estimate
    let addedDepth = len(subtreeNodes) div 2
    if currentDepth + addedDepth > maxDepth:
      # Would exceed depth, try again or skip
      continue

    # Replace subtree in offspring
    replaceSubtree(offspring, targetIdx, subtreeNodes)

    # Success
    return offspring

  # All attempts exceeded depth limit, return parent1 unchanged
  return parent1


# ============================================================================
# Mutation Subtree Generator (Standalone to avoid closure issues)
# ============================================================================

proc generateMutationNode(
  rng: var Rand,
  depth: int,
  maxAllowedDepth: int,
  numFeatures: int,
  availableOps: seq[OperationKind],
  nodes: var seq[StackProgramNode]
): int =
  ## Helper to generate a single node for mutation
  ## Returns the index of the created node

  let leafProbability = depth / maxAllowedDepth
  let forceInternal = (depth == 0)

  if (not forceInternal) and (rng.rand(1.0) < leafProbability or depth >= maxAllowedDepth):
    # Create leaf node (feature)
    let maxFeatureIdx = max(0, numFeatures - 1)
    let featureIdx = rng.rand(maxFeatureIdx)
    let nodeIdx = len(nodes)
    nodes.add(StackProgramNode(
      left: -1,
      right: -1,
      kind: opFeature,
      featureIndex: featureIdx
    ))
    return nodeIdx
  else:
    # Create internal node (operation)
    var unaryOps = newSeq[OperationKind]()
    var binaryOps = newSeq[OperationKind]()
    for op in availableOps:
      if op in {opNegate, opSquare, opCube, opSin, opCos, opTan, opSqrt, opAbs}:
        unaryOps.add(op)
      elif op in {opAddConstant, opMulConstant}:
        unaryOps.add(op)
      else:
        binaryOps.add(op)

    var selectedOp: OperationKind
    if len(unaryOps) > 0 and len(binaryOps) > 0:
      if rng.rand(1.0) < 0.5:
        let maxIdx = max(0, high(unaryOps))
        selectedOp = unaryOps[rng.rand(maxIdx)]
      else:
        let maxIdx = max(0, high(binaryOps))
        selectedOp = binaryOps[rng.rand(maxIdx)]
    elif len(unaryOps) > 0:
      let maxIdx = max(0, high(unaryOps))
      selectedOp = unaryOps[rng.rand(maxIdx)]
    else:
      let maxIdx = max(0, high(binaryOps))
      selectedOp = binaryOps[rng.rand(maxIdx)]

    let isUnary = selectedOp in {opNegate, opSquare, opCube, opSin, opCos, opTan, opSqrt, opAbs}
    let isConstant = selectedOp in {opAddConstant, opMulConstant}

    if isUnary:
      let childIdx = generateMutationNode(rng, depth + 1, maxAllowedDepth, numFeatures, availableOps, nodes)
      let nodeIdx = len(nodes)
      nodes.add(StackProgramNode(
        left: childIdx,
        right: -1,
        kind: selectedOp
      ))
      return nodeIdx
    elif isConstant:
      let childIdx = generateMutationNode(rng, depth + 1, maxAllowedDepth, numFeatures, availableOps, nodes)
      let constant = rng.rand(1.0) * 2.0 - 1.0
      let opKind = selectedOp

      case opKind
      of opAddConstant:
        nodes.add(StackProgramNode(
          left: childIdx,
          right: -1,
          kind: opKind,
          addConstantValue: constant
        ))
      of opMulConstant:
        nodes.add(StackProgramNode(
          left: childIdx,
          right: -1,
          kind: opKind,
          mulConstantValue: constant
        ))
      else:
        discard

      return len(nodes) - 1
    else:
      # Binary operation
      let leftIdx = generateMutationNode(rng, depth + 1, maxAllowedDepth, numFeatures, availableOps, nodes)
      let rightIdx = generateMutationNode(rng, depth + 1, maxAllowedDepth, numFeatures, availableOps, nodes)
      let nodeIdx = len(nodes)
      nodes.add(StackProgramNode(
        left: leftIdx,
        right: rightIdx,
        kind: selectedOp
      ))
      return nodeIdx


proc generateRandomSubtree(
  rng: var Rand,
  maxDepth: int,
  numFeatures: int,
  availableOps: seq[OperationKind]
): seq[StackProgramNode] =
  ## Generate a random subtree for mutation
  var nodes = newSeq[StackProgramNode](0)
  discard generateMutationNode(rng, 0, maxDepth, numFeatures, availableOps, nodes)
  return nodes


proc mutate*(program: StackProgram, rng: var Rand, maxDepth: int, numFeatures: int, availableOps: seq[OperationKind]): StackProgram =
  ## Perform subtree mutation on a program
  ##
  ## Strategy:
  ## 1. Select random subtree in program
  ## 2. Generate new random subtree
  ## 3. Replace selected subtree with new one IF depth limit not exceeded
  ##
  ## This creates local diversity while preserving most of the program structure

  if len(program.nodes) == 0:
    return program

  # Clone program to modify
  var offspring = StackProgram(nodes: program.nodes, depth: program.depth)

  # Try mutation up to 3 times to find one that doesn't exceed depth limit
  for attempt in 0..<3:
    # Select random mutation point
    let targetIdx = getRandomNodeIndex(offspring, rng)
    if targetIdx < 0:
      break

    # Find current depth at target node (approximate)
    let currentDepth = min(targetIdx div 2, maxDepth - 2)  # Rough estimate, leave room for subtree

    # Calculate remaining depth for the new subtree
    let remainingDepth = max(1, maxDepth - currentDepth)

    # Generate new random subtree using standalone helper
    var nodes = generateRandomSubtree(rng, remainingDepth, numFeatures, availableOps)

    # Check if replacement would exceed max depth
    let oldSize = findSubtreeBounds(offspring, targetIdx).size
    let offspringDepth = len(offspring.nodes) div 2
    let newDepth = len(nodes) div 2
    if offspringDepth - oldSize + newDepth > maxDepth:
      # Would exceed depth, try again or skip
      continue

    # Replace subtree in offspring
    replaceSubtree(offspring, targetIdx, nodes)

    # Success
    return offspring

  # All attempts failed, return original
  return program


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
    # Value type cannot return nil - return empty program instead
    return StackProgram(nodes: @[], depth: 0)

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
export countNodes, findSubtreeBounds, getRandomNodeIndex, cloneSubtree, replaceSubtree,
       crossover, mutate, tournamentSelect, generateMutationNode, generateRandomSubtree
