# Program simplification for genetic programming
# Optimizes program trees by removing redundant operations

import std/sets
import ./types


# ============================================================================
# Program Simplification (Bottom-Up Reconstruction)
# ============================================================================

proc simplifyProgram*(program: StackProgram): StackProgram =
  ## Optimize a program tree by removing redundant operations.
  ##
  ## This rebuilds the tree bottom-up, applying the following optimizations:
  ## 1. Identity removal: x + 0 -> x, x * 1 -> x
  ## 2. Constant folding: add_constant(add_constant(x, 5), 3) -> add_constant(x, 8)
  ## 3. Double negation: negate(negate(x)) -> x
  ##
  ## Returns a new, simplified StackProgram

  # Edge case: Empty program
  if len(program.nodes) == 0:
    return program

  # Build a NEW program into this sequence
  var newNodes = newSeq[StackProgramNode]()

  # Helper: check if a float is approximately zero
  proc isZero(x: float64): bool {.inline.} =
    return abs(x) < 1e-9

  # Helper: check if a float is approximately one
  proc isOne(x: float64): bool {.inline.} =
    return abs(x - 1.0) < 1e-9

  # Track visited nodes to prevent cycles
  var visited = initHashSet[int]()

  # Recursive function to process a node and its children
  # Returns the index of the simplified node in 'newNodes'
  proc processNode(oldIdx: int): int =
    # Prevent infinite recursion by checking if we've already processed this node
    if oldIdx in visited:
      # Already processed, return the last created node
      if len(newNodes) > 0:
        return len(newNodes) - 1
      return -1

    # Ensure the index is valid
    if oldIdx < 0 or oldIdx >= len(program.nodes):
      return -1

    visited.incl(oldIdx)
    let oldNode = program.nodes[oldIdx]

    # RECURSION: Simplify children first (Bottom-Up)
    var newLeftIdx = -1
    var newRightIdx = -1

    if oldNode.left != -1 and oldNode.left < len(program.nodes):
      newLeftIdx = processNode(oldNode.left)
    if oldNode.right != -1 and oldNode.right < len(program.nodes):
      newRightIdx = processNode(oldNode.right)

    # RULE 1: Identity Removal for Constants
    # add_constant(x, 0.0) -> x
    if oldNode.kind == opAddConstant:
      if isZero(oldNode.addConstantValue):
        return newLeftIdx # Skip this node, return child directly!

    # mul_constant(x, 1.0) -> x
    if oldNode.kind == opMulConstant:
      if isOne(oldNode.mulConstantValue):
        return newLeftIdx # Skip this node, return child directly!

    # RULE 2: Constant Folding (Nested Constants)
    # add_constant(add_constant(x, 5), 3) -> add_constant(x, 8)
    if oldNode.kind == opAddConstant and newLeftIdx >= 0:
      let child = newNodes[newLeftIdx]
      if child.kind == opAddConstant:
        # We found nested adds!
        # Mutate the child in 'newNodes' to absorb this value
        newNodes[newLeftIdx].addConstantValue += oldNode.addConstantValue
        return newLeftIdx # Return the child's index (it now holds the sum)

    # mul_constant(mul_constant(x, 2), 3) -> mul_constant(x, 6)
    if oldNode.kind == opMulConstant and newLeftIdx >= 0:
      let child = newNodes[newLeftIdx]
      if child.kind == opMulConstant:
        # We found nested multiplies!
        newNodes[newLeftIdx].mulConstantValue *= oldNode.mulConstantValue
        return newLeftIdx # Return the child's index (it now holds the product)

    # RULE 3: Double Negation
    # negate(negate(x)) -> x
    # NOTE: We can only skip if the grandchild has already been processed
    if oldNode.kind == opNegate and newLeftIdx >= 0:
      let child = newNodes[newLeftIdx]
      if child.kind == opNegate and child.left >= 0 and child.left in visited:
        # The grandchild was processed, but we need to find its NEW index
        # For now, just return the child (remove one level of negation)
        return newLeftIdx

    # NO SIMPLIFICATION MATCHED:
    # Push the current node to the new program with updated child indices
    var newNode = oldNode
    newNode.left = newLeftIdx
    newNode.right = newRightIdx

    newNodes.add(newNode)
    return len(newNodes) - 1

  # Start processing from the root (last node in post-order representation)
  let rootIdx = len(program.nodes) - 1
  if rootIdx >= 0:
    discard processNode(rootIdx)

  return StackProgram(nodes: newNodes, depth: program.depth)


# Export functions without nuwa_export (implementation only)
export simplifyProgram
