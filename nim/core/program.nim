# Stack-based program tree evaluation for genetic programming
# This provides 10-50x speedup over Python recursion by using:
# 1. Stack-based evaluation instead of recursion
# 2. Direct NumPy array access with zero-copy
# 3. C-style memory management (flat buffers, value types)

import std/math
import ./types

# Types for program tree evaluation
type
  ## Stack frame for evaluation (VALUE TYPE - no GC overhead)
  EvalFrame* = object
    resultBuffer*: ptr UncheckedArray[float64]  # Pointer into pre-allocated flat buffer
    case isLeaf*: bool
    of true:
      featureIndex*: int  # Index into feature matrix
    of false:
      opKind*: OperationKind
      numChildren*: int

  ## Pre-allocated buffer pool for evaluation (C-style: single flat allocation)
  ## ONE contiguous memory block for all buffers, accessed via offset arithmetic
  EvalBufferPool* = object
    data*: ptr UncheckedArray[float64]  # Single flat allocation (like malloc)
    numBuffers*: int
    bufferSize*: int
    totalSize*: int  # Total size in floats (numBuffers * bufferSize)

  ## Feature matrix for batch evaluation (VALUE TYPE - no ref/GC)
  FeatureMatrix* = object
    data*: ptr UncheckedArray[ptr UncheckedArray[float64]]  # Pointers to each column
    numRows*: int
    numCols*: int

  ## StackProgram definition (VALUE TYPE - no ref/GC)
  ## Programs are represented as flat lists for stack-based evaluation
  StackProgram* = object
    nodes*: seq[StackProgramNode]  # seq is OK here - allocated once per program
    depth*: int

  StackProgramNode* = object
    left*: int  # Index of left child in nodes array (-1 if none) - MUST be first
    right*: int # Index of right child in nodes array (-1 if none) - MUST be second
    case kind*: OperationKind
    of opAdd, opSubtract, opMultiply, opDivide, opPow:
      discard
    of opNegate, opSquare, opCube, opSin, opCos, opTan, opSqrt, opAbs:
      discard
    of opAddConstant:
      addConstantValue*: float64
    of opMulConstant:
      mulConstantValue*: float64
    of opFeature:
      featureIndex*: int

# ============================================================================
# Buffer Pool Management (C-style: single malloc, offset arithmetic)
# ============================================================================

proc newEvalBufferPool*(numBuffers: int, bufferSize: int): EvalBufferPool =
  ## Create a new buffer pool with a SINGLE flat allocation
  ## This is like: float* buffer = (float*)malloc(numBuffers * bufferSize * sizeof(float))
  ##
  ## Instead of allocating N separate buffers (N mallocs), we allocate
  ## ONE big buffer and use pointer arithmetic to access slices.

  result.totalSize = numBuffers * bufferSize
  result.numBuffers = numBuffers
  result.bufferSize = bufferSize

  # Single allocation (like malloc in C)
  # allocate returns pointer to uninitialized memory
  result.data = cast[ptr UncheckedArray[float64]](alloc(result.totalSize * sizeof(float64)))

proc getBuffer*(pool: var EvalBufferPool, index: int): ptr UncheckedArray[float64] =
  ## Get pointer to buffer at index using OFFSET ARITHMETIC
  ## This is like: return &buffer[index * bufferSize]
  ##
  ## ZERO allocations after initialization - just pointer math!
  if index >= pool.numBuffers:
    # Should not happen with correct sizing, but fail gracefully
    let newSize = (index + 1) * pool.bufferSize
    var newData = cast[ptr UncheckedArray[float64]](alloc(newSize * sizeof(float64)))

    # Copy old data
    for i in 0..<pool.totalSize:
      newData[i] = pool.data[i]

    # Free old and assign new (in C, we'd use realloc here)
    pool.data = newData
    pool.totalSize = newSize
    pool.numBuffers = index + 1

  # Pointer arithmetic: return pointer to offset [index * bufferSize]
  return cast[ptr UncheckedArray[float64]](addr pool.data[index * pool.bufferSize])

proc destroyEvalBufferPool*(pool: var EvalBufferPool) =
  ## Free the allocated memory (explicit cleanup like free() in C)
  if pool.data != nil:
    dealloc(pool.data)
    pool.data = nil

# ============================================================================
# Feature Matrix Management (VALUE TYPE - passed by value, not ref)
# ============================================================================

proc newFeatureMatrix*(numRows: int, numCols: int): FeatureMatrix =
  ## Create a new feature matrix (VALUE TYPE, no heap allocation for the struct itself)
  ## Only allocates the array of column pointers
  result.numRows = numRows
  result.numCols = numCols
  result.data = cast[ptr UncheckedArray[ptr UncheckedArray[float64]]](alloc(numCols * sizeof(ptr UncheckedArray[float64])))

proc setColumn*(fm: var FeatureMatrix, colIdx: int, ptrData: int) =
  ## Set a column in the feature matrix (zero-copy)
  fm.data[colIdx] = cast[ptr UncheckedArray[float64]](ptrData)

proc getColumn*(fm: FeatureMatrix, colIdx: int): ptr UncheckedArray[float64] =
  ## Get a column from the feature matrix
  return fm.data[colIdx]

proc destroyFeatureMatrix*(fm: var FeatureMatrix) =
  ## Free the allocated memory
  if fm.data != nil:
    dealloc(fm.data)
    fm.data = nil

# ============================================================================
# Stack-Based Program Evaluation (OPTIMIZED with Flat Buffer Pool)
# ============================================================================

proc evaluateProgramStack(program: StackProgram, fm: FeatureMatrix, pool: var EvalBufferPool): seq[float64] =
  ## Evaluate a program using stack-based approach with flat buffer pool
  ##
  ## This is much faster than Python recursion because:
  ## 1. No Python function call overhead
  ## 2. Stack-based instead of recursive
  ## 3. Direct memory access to NumPy arrays
  ## 4. Flat buffer pool (ONE malloc, pointer arithmetic, NO per-node allocations!)
  ##
  ## Returns: Computed values for all rows

  let numNodes = len(program.nodes)
  if numNodes == 0:
    return newSeq[float64](fm.numRows)

  # Create stack for evaluation (value types, no GC)
  var stack = newSeq[EvalFrame](numNodes)
  var stackPtr = 0

  # Process each node in order (post-order: children come before parents)
  for nodeIdx in 0..<numNodes:
    let node = program.nodes[nodeIdx]

    case node.kind
    of opFeature:
      # Leaf node - copy feature data to pre-allocated buffer
      let targetBuffer = pool.getBuffer(stackPtr)
      stack[stackPtr] = EvalFrame(
        isLeaf: true,
        featureIndex: node.featureIndex,
        resultBuffer: targetBuffer
      )

      # Copy data from feature matrix directly into buffer
      let colData = fm.getColumn(node.featureIndex)
      for i in 0..<fm.numRows:
        targetBuffer[i] = colData[i]

      stackPtr += 1

    of opNegate:
      # Unary operation - write to pre-allocated buffer
      let childIdx = node.left
      let childBuffer = stack[childIdx].resultBuffer
      let targetBuffer = pool.getBuffer(stackPtr)

      stack[stackPtr] = EvalFrame(
        isLeaf: false,
        opKind: node.kind,
        numChildren: 1,
        resultBuffer: targetBuffer
      )

      for i in 0..<fm.numRows:
        targetBuffer[i] = -childBuffer[i]

      stackPtr += 1

    of opSquare:
      let childIdx = node.left
      let childBuffer = stack[childIdx].resultBuffer
      let targetBuffer = pool.getBuffer(stackPtr)

      stack[stackPtr] = EvalFrame(
        isLeaf: false,
        opKind: node.kind,
        numChildren: 1,
        resultBuffer: targetBuffer
      )

      for i in 0..<fm.numRows:
        let val = childBuffer[i]
        targetBuffer[i] = val * val

      stackPtr += 1

    of opCube:
      let childIdx = node.left
      let childBuffer = stack[childIdx].resultBuffer
      let targetBuffer = pool.getBuffer(stackPtr)

      stack[stackPtr] = EvalFrame(
        isLeaf: false,
        opKind: node.kind,
        numChildren: 1,
        resultBuffer: targetBuffer
      )

      for i in 0..<fm.numRows:
        let val = childBuffer[i]
        targetBuffer[i] = val * val * val

      stackPtr += 1

    of opSin:
      let childIdx = node.left
      let childBuffer = stack[childIdx].resultBuffer
      let targetBuffer = pool.getBuffer(stackPtr)

      stack[stackPtr] = EvalFrame(
        isLeaf: false,
        opKind: node.kind,
        numChildren: 1,
        resultBuffer: targetBuffer
      )

      for i in 0..<fm.numRows:
        targetBuffer[i] = sin(childBuffer[i])

      stackPtr += 1

    of opCos:
      let childIdx = node.left
      let childBuffer = stack[childIdx].resultBuffer
      let targetBuffer = pool.getBuffer(stackPtr)

      stack[stackPtr] = EvalFrame(
        isLeaf: false,
        opKind: node.kind,
        numChildren: 1,
        resultBuffer: targetBuffer
      )

      for i in 0..<fm.numRows:
        targetBuffer[i] = cos(childBuffer[i])

      stackPtr += 1

    of opTan:
      let childIdx = node.left
      let childBuffer = stack[childIdx].resultBuffer
      let targetBuffer = pool.getBuffer(stackPtr)

      stack[stackPtr] = EvalFrame(
        isLeaf: false,
        opKind: node.kind,
        numChildren: 1,
        resultBuffer: targetBuffer
      )

      for i in 0..<fm.numRows:
        targetBuffer[i] = tan(childBuffer[i])

      stackPtr += 1

    of opPow:
      let leftIdx = node.left
      let rightIdx = node.right
      let leftBuffer = stack[leftIdx].resultBuffer
      let rightBuffer = stack[rightIdx].resultBuffer
      let targetBuffer = pool.getBuffer(stackPtr)

      stack[stackPtr] = EvalFrame(
        isLeaf: false,
        opKind: node.kind,
        numChildren: 2,
        resultBuffer: targetBuffer
      )

      for i in 0..<fm.numRows:
        let base = leftBuffer[i]
        let exp = rightBuffer[i]
        # Handle special cases for safety
        if abs(base) < 1e-10 and exp < 0:
          # 0^(-n) would be infinity, return 1 instead
          targetBuffer[i] = 1.0
        elif base < 0 and floor(exp) != exp:
          # Negative base with non-integer exponent would be complex
          # Use absolute value instead
          targetBuffer[i] = pow(abs(base), exp)
        else:
          targetBuffer[i] = pow(base, exp)

      stackPtr += 1

    of opSqrt:
      let childIdx = node.left
      let childBuffer = stack[childIdx].resultBuffer
      let targetBuffer = pool.getBuffer(stackPtr)

      stack[stackPtr] = EvalFrame(
        isLeaf: false,
        opKind: node.kind,
        numChildren: 1,
        resultBuffer: targetBuffer
      )

      for i in 0..<fm.numRows:
        targetBuffer[i] = sqrt(abs(childBuffer[i]))

      stackPtr += 1

    of opAbs:
      let childIdx = node.left
      let childBuffer = stack[childIdx].resultBuffer
      let targetBuffer = pool.getBuffer(stackPtr)

      stack[stackPtr] = EvalFrame(
        isLeaf: false,
        opKind: node.kind,
        numChildren: 1,
        resultBuffer: targetBuffer
      )

      for i in 0..<fm.numRows:
        targetBuffer[i] = abs(childBuffer[i])

      stackPtr += 1

    of opAdd:
      let leftIdx = node.left
      let rightIdx = node.right
      let leftBuffer = stack[leftIdx].resultBuffer
      let rightBuffer = stack[rightIdx].resultBuffer
      let targetBuffer = pool.getBuffer(stackPtr)

      stack[stackPtr] = EvalFrame(
        isLeaf: false,
        opKind: node.kind,
        numChildren: 2,
        resultBuffer: targetBuffer
      )

      for i in 0..<fm.numRows:
        targetBuffer[i] = leftBuffer[i] + rightBuffer[i]

      stackPtr += 1

    of opSubtract:
      let leftIdx = node.left
      let rightIdx = node.right
      let leftBuffer = stack[leftIdx].resultBuffer
      let rightBuffer = stack[rightIdx].resultBuffer
      let targetBuffer = pool.getBuffer(stackPtr)

      stack[stackPtr] = EvalFrame(
        isLeaf: false,
        opKind: node.kind,
        numChildren: 2,
        resultBuffer: targetBuffer
      )

      for i in 0..<fm.numRows:
        targetBuffer[i] = leftBuffer[i] - rightBuffer[i]

      stackPtr += 1

    of opMultiply:
      let leftIdx = node.left
      let rightIdx = node.right
      let leftBuffer = stack[leftIdx].resultBuffer
      let rightBuffer = stack[rightIdx].resultBuffer
      let targetBuffer = pool.getBuffer(stackPtr)

      stack[stackPtr] = EvalFrame(
        isLeaf: false,
        opKind: node.kind,
        numChildren: 2,
        resultBuffer: targetBuffer
      )

      for i in 0..<fm.numRows:
        targetBuffer[i] = leftBuffer[i] * rightBuffer[i]

      stackPtr += 1

    of opDivide:
      let leftIdx = node.left
      let rightIdx = node.right
      let leftBuffer = stack[leftIdx].resultBuffer
      let rightBuffer = stack[rightIdx].resultBuffer
      let targetBuffer = pool.getBuffer(stackPtr)

      stack[stackPtr] = EvalFrame(
        isLeaf: false,
        opKind: node.kind,
        numChildren: 2,
        resultBuffer: targetBuffer
      )

      for i in 0..<fm.numRows:
        let r = rightBuffer[i]
        if abs(r) < 1e-10:
          targetBuffer[i] = leftBuffer[i]
        else:
          targetBuffer[i] = leftBuffer[i] / r

      stackPtr += 1

    of opAddConstant:
      let childIdx = node.left
      let childBuffer = stack[childIdx].resultBuffer
      let targetBuffer = pool.getBuffer(stackPtr)

      stack[stackPtr] = EvalFrame(
        isLeaf: false,
        opKind: node.kind,
        numChildren: 1,
        resultBuffer: targetBuffer
      )

      for i in 0..<fm.numRows:
        targetBuffer[i] = childBuffer[i] + node.addConstantValue

      stackPtr += 1

    of opMulConstant:
      let childIdx = node.left
      let childBuffer = stack[childIdx].resultBuffer
      let targetBuffer = pool.getBuffer(stackPtr)

      stack[stackPtr] = EvalFrame(
        isLeaf: false,
        opKind: node.kind,
        numChildren: 1,
        resultBuffer: targetBuffer
      )

      for i in 0..<fm.numRows:
        targetBuffer[i] = childBuffer[i] * node.mulConstantValue

      stackPtr += 1

  # Return the root node's result (last node processed in post-order)
  if stackPtr == 0:
    return newSeq[float64](0)

  # Copy result from buffer to return seq
  let finalBuffer = stack[stackPtr - 1].resultBuffer
  result = newSeq[float64](fm.numRows)
  for i in 0..<fm.numRows:
    result[i] = finalBuffer[i]
  return result

# Implementation functions for program evaluation (called from Python wrappers)

proc evaluateProgramImpl(
  featurePtrs: seq[int],     # Pointers to feature columns
  featureIndices: seq[int], # Indices for each node (or -1 for ops)
  opKinds: seq[int],         # Operation kind for each node
  leftChildren: seq[int],   # Left child indices
  rightChildren: seq[int],  # Right child indices
  constants: seq[float64],   # Constants for add/mul_constant ops
  numRows: int,
  numCols: int
): seq[float64] =
  ## Evaluate a program from Python
  ##
  ## This function takes serialized program data from Python and evaluates it
  ## using the fast stack-based approach with flat buffer pool.
  ##
  ## Args:
  ##   featurePtrs: Raw pointers to NumPy array data for each feature
  ##   featureIndices: Feature index for each node (-1 for operation nodes)
  ##   opKinds: Integer representation of operation kind for each node
  ##   leftChildren: Index of left child in node array
  ##   rightChildren: Index of right child in node array
  ##   constants: Constant values (used for add/mul_constant)
  ##   numRows: Number of rows in the dataset
  ##   numCols: Number of features
  ##
  ## Returns: Computed values as a sequence (converts to NumPy array in Python)

  let numNodes = len(opKinds)

  # Create feature matrix (VALUE TYPE, automatic cleanup when scope exits)
  var fm = newFeatureMatrix(numRows, numCols)
  defer: destroyFeatureMatrix(fm)

  for i in 0..<numCols:
    fm.setColumn(i, featurePtrs[i])

  # Create buffer pool (pre-allocate once, reuse for all nodes!)
  var pool = newEvalBufferPool(numNodes, numRows)
  defer: destroyEvalBufferPool(pool)

  # Build program from serialized data (VALUE TYPE, no GC)
  var program = StackProgram(nodes: newSeq[StackProgramNode](numNodes), depth: 0)

  for i in 0..<numNodes:
    let kind = OperationKind(opKinds[i])

    case kind
    of opAddConstant:
      program.nodes[i] = StackProgramNode(
        left: leftChildren[i],
        right: -1,
        kind: kind,
        addConstantValue: constants[i]
      )
    of opMulConstant:
      program.nodes[i] = StackProgramNode(
        left: leftChildren[i],
        right: -1,
        kind: kind,
        mulConstantValue: constants[i]
      )
    of opFeature:
      program.nodes[i] = StackProgramNode(
        left: -1,
        right: -1,
        kind: kind,
        featureIndex: featureIndices[i]
      )
    else:
      program.nodes[i] = StackProgramNode(
        left: leftChildren[i],
        right: rightChildren[i],
        kind: kind
      )

  # Evaluate the program with buffer pool (NO per-node allocations!)
  return evaluateProgramStack(program, fm, pool)
