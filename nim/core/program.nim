# Stack evaluation, buffer pool, and FeatureMatrix (column views into NumPy)

import std/math

type
  EvalFrame* = object
    resultBuffer*: ptr UncheckedArray[float64]
    case isLeaf*: bool
    of true:
      featureIndex*: int
    of false:
      opKind*: OperationKind
      numChildren*: int

  EvalBufferPool* = object
    data*: ptr UncheckedArray[float64]
    numBuffers*: int
    bufferSize*: int
    totalSize*: int

  FeatureMatrix* = object
    data*: ptr UncheckedArray[ptr UncheckedArray[float64]]
    numRows*: int
    numCols*: int

  StackProgram* = object
    nodes*: seq[StackProgramNode]
    depth*: int

  StackProgramNode* = object
    left*: int
    right*: int
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

  SerializedProgram* = tuple[
    featureIndices: seq[int],
    opKinds: seq[int],
    leftChildren: seq[int],
    rightChildren: seq[int],
    constants: seq[float64]
  ]

proc newEvalBufferPool*(numBuffers: int, bufferSize: int): EvalBufferPool =
  result.totalSize = numBuffers * bufferSize
  result.numBuffers = numBuffers
  result.bufferSize = bufferSize
  result.data = cast[ptr UncheckedArray[float64]](alloc(result.totalSize *
      sizeof(float64)))

proc getBuffer*(pool: var EvalBufferPool, index: int): ptr UncheckedArray[float64] =
  if index >= pool.numBuffers:
    let newSize = (index + 1) * pool.bufferSize
    var newData = cast[ptr UncheckedArray[float64]](alloc(newSize * sizeof(float64)))
    for i in 0..<pool.totalSize:
      newData[i] = pool.data[i]
    if pool.data != nil:
      dealloc(pool.data)
    pool.data = newData
    pool.totalSize = newSize
    pool.numBuffers = index + 1
  return cast[ptr UncheckedArray[float64]](addr pool.data[index * pool.bufferSize])

proc destroyEvalBufferPool*(pool: var EvalBufferPool) =
  if pool.data != nil:
    dealloc(pool.data)
    pool.data = nil

proc newFeatureMatrix*(numRows: int, numCols: int): FeatureMatrix =
  result.numRows = numRows
  result.numCols = numCols
  result.data = cast[ptr UncheckedArray[ptr UncheckedArray[float64]]](alloc(
      numCols * sizeof(ptr UncheckedArray[float64])))

proc setColumn*(fm: var FeatureMatrix, colIdx: int,
    col: ptr UncheckedArray[float64]) =
  fm.data[colIdx] = col

proc getColumn*(fm: FeatureMatrix, colIdx: int): ptr UncheckedArray[float64] =
  fm.data[colIdx]

proc cloneFeatureMatrix*(fm: FeatureMatrix): FeatureMatrix =
  ## New column-pointer array, same underlying NumPy columns (thread-local wrapper).
  result = newFeatureMatrix(fm.numRows, fm.numCols)
  for i in 0..<fm.numCols:
    result.data[i] = fm.data[i]

proc destroyFeatureMatrix*(fm: var FeatureMatrix) =
  if fm.data != nil:
    dealloc(fm.data)
    fm.data = nil

proc stackProgramFromSerialized*(
  featureIndices: seq[int],
  opKinds: seq[int],
  leftChildren: seq[int],
  rightChildren: seq[int],
  constants: seq[float64]
): StackProgram =
  let numNodes = len(opKinds)
  result = StackProgram(nodes: newSeq[StackProgramNode](numNodes), depth: 0)
  for i in 0..<numNodes:
    let kind = OperationKind(opKinds[i])
    case kind
    of opAddConstant:
      result.nodes[i] = StackProgramNode(
        left: leftChildren[i], right: -1, kind: kind,
        addConstantValue: constants[i]
      )
    of opMulConstant:
      result.nodes[i] = StackProgramNode(
        left: leftChildren[i], right: -1, kind: kind,
        mulConstantValue: constants[i]
      )
    of opFeature:
      result.nodes[i] = StackProgramNode(
        left: -1, right: -1, kind: kind, featureIndex: featureIndices[i]
      )
    else:
      result.nodes[i] = StackProgramNode(
        left: leftChildren[i], right: rightChildren[i], kind: kind
      )

proc serializeStackProgram*(program: StackProgram): SerializedProgram =
  let n = len(program.nodes)
  result.featureIndices = newSeq[int](n)
  result.opKinds = newSeq[int](n)
  result.leftChildren = newSeq[int](n)
  result.rightChildren = newSeq[int](n)
  result.constants = newSeq[float64](n)
  for i, node in program.nodes:
    result.featureIndices[i] = if node.kind == opFeature: node.featureIndex else: -1
    result.opKinds[i] = ord(node.kind)
    result.leftChildren[i] = node.left
    result.rightChildren[i] = node.right
    if node.kind == opAddConstant:
      result.constants[i] = node.addConstantValue
    elif node.kind == opMulConstant:
      result.constants[i] = node.mulConstantValue
    else:
      result.constants[i] = 0.0

proc evaluateProgramStack(program: StackProgram, fm: FeatureMatrix,
    pool: var EvalBufferPool): seq[float64] =
  let numNodes = len(program.nodes)
  if numNodes == 0:
    return newSeq[float64](fm.numRows)

  var stack = newSeq[EvalFrame](numNodes)
  var stackPtr = 0

  template pushUnary(body: untyped) {.dirty.} =
    let childBuffer {.inject.} = stack[node.left].resultBuffer
    let targetBuffer {.inject.} = pool.getBuffer(stackPtr)
    stack[stackPtr] = EvalFrame(
      isLeaf: false, opKind: node.kind, numChildren: 1, resultBuffer: targetBuffer
    )
    for i in 0..<fm.numRows:
      body
    stackPtr += 1

  template pushBinary(body: untyped) {.dirty.} =
    let leftBuffer {.inject.} = stack[node.left].resultBuffer
    let rightBuffer {.inject.} = stack[node.right].resultBuffer
    let targetBuffer {.inject.} = pool.getBuffer(stackPtr)
    stack[stackPtr] = EvalFrame(
      isLeaf: false, opKind: node.kind, numChildren: 2, resultBuffer: targetBuffer
    )
    for i in 0..<fm.numRows:
      body
    stackPtr += 1

  for nodeIdx in 0..<numNodes:
    let node = program.nodes[nodeIdx]
    case node.kind
    of opFeature:
      stack[stackPtr] = EvalFrame(
        isLeaf: true,
        featureIndex: node.featureIndex,
        resultBuffer: fm.getColumn(node.featureIndex),
      )
      stackPtr += 1
    of opNegate:
      pushUnary:
        targetBuffer[i] = -childBuffer[i]
    of opSquare:
      pushUnary:
        let val = childBuffer[i]
        targetBuffer[i] = val * val
    of opCube:
      pushUnary:
        let val = childBuffer[i]
        targetBuffer[i] = val * val * val
    of opSin:
      pushUnary:
        targetBuffer[i] = sin(childBuffer[i])
    of opCos:
      pushUnary:
        targetBuffer[i] = cos(childBuffer[i])
    of opTan:
      pushUnary:
        targetBuffer[i] = tan(childBuffer[i])
    of opSqrt:
      pushUnary:
        targetBuffer[i] = sqrt(abs(childBuffer[i]))
    of opAbs:
      pushUnary:
        targetBuffer[i] = abs(childBuffer[i])
    of opAddConstant:
      pushUnary:
        targetBuffer[i] = childBuffer[i] + node.addConstantValue
    of opMulConstant:
      pushUnary:
        targetBuffer[i] = childBuffer[i] * node.mulConstantValue
    of opAdd:
      pushBinary:
        targetBuffer[i] = leftBuffer[i] + rightBuffer[i]
    of opSubtract:
      pushBinary:
        targetBuffer[i] = leftBuffer[i] - rightBuffer[i]
    of opMultiply:
      pushBinary:
        targetBuffer[i] = leftBuffer[i] * rightBuffer[i]
    of opDivide:
      pushBinary:
        let r = rightBuffer[i]
        if abs(r) < 1e-10:
          targetBuffer[i] = leftBuffer[i]
        else:
          targetBuffer[i] = leftBuffer[i] / r
    of opPow:
      pushBinary:
        let base = leftBuffer[i]
        let exp = rightBuffer[i]
        if abs(base) < 1e-10 and exp < 0:
          targetBuffer[i] = 1.0
        elif base < 0 and floor(exp) != exp:
          targetBuffer[i] = pow(abs(base), exp)
        else:
          targetBuffer[i] = pow(base, exp)

  if stackPtr == 0:
    return newSeq[float64](0)
  let finalBuffer = stack[stackPtr - 1].resultBuffer
  result = newSeq[float64](fm.numRows)
  for i in 0..<fm.numRows:
    result[i] = finalBuffer[i]

proc evaluateProgramImpl*(
  fm: FeatureMatrix,
  featureIndices: seq[int],
  opKinds: seq[int],
  leftChildren: seq[int],
  rightChildren: seq[int],
  constants: seq[float64]
): seq[float64] =
  let numNodes = len(opKinds)
  var pool = newEvalBufferPool(numNodes, fm.numRows)
  defer: destroyEvalBufferPool(pool)
  let program = stackProgramFromSerialized(
    featureIndices, opKinds, leftChildren, rightChildren, constants
  )
  evaluateProgramStack(program, fm, pool)

proc evaluateProgramsBatchedImpl*(
  fm: FeatureMatrix,
  programSizes: seq[int],
  featureIndicesFlat: seq[int],
  opKindsFlat: seq[int],
  leftChildrenFlat: seq[int],
  rightChildrenFlat: seq[int],
  constantsFlat: seq[float64]
): seq[seq[float64]] =
  let numPrograms = len(programSizes)
  result = newSeq[seq[float64]](numPrograms)
  var offset = 0
  for i in 0..<numPrograms:
    let progSize = programSizes[i]
    var featureIndices = newSeq[int](progSize)
    var opKinds = newSeq[int](progSize)
    var leftChildren = newSeq[int](progSize)
    var rightChildren = newSeq[int](progSize)
    var constants = newSeq[float64](progSize)
    for j in 0..<progSize:
      let idx = offset + j
      featureIndices[j] = featureIndicesFlat[idx]
      opKinds[j] = opKindsFlat[idx]
      leftChildren[j] = leftChildrenFlat[idx]
      rightChildren[j] = rightChildrenFlat[idx]
      constants[j] = constantsFlat[idx]
    result[i] = evaluateProgramImpl(
      fm, featureIndices, opKinds, leftChildren, rightChildren, constants
    )
    offset += progSize

proc serializePopulation*(pop: seq[StackProgram]): tuple[
  programSizes: seq[int],
  featureIndicesFlat: seq[int],
  opKindsFlat: seq[int],
  leftChildrenFlat: seq[int],
  rightChildrenFlat: seq[int],
  constantsFlat: seq[float64]
] =
  result.programSizes = newSeq[int](pop.len)
  for i, program in pop:
    let ser = serializeStackProgram(program)
    result.programSizes[i] = len(ser.opKinds)
    result.featureIndicesFlat.add(ser.featureIndices)
    result.opKindsFlat.add(ser.opKinds)
    result.leftChildrenFlat.add(ser.leftChildren)
    result.rightChildrenFlat.add(ser.rightChildren)
    result.constantsFlat.add(ser.constants)

proc deserializePopulation*(
  programSizes: seq[int],
  featureIndicesFlat: seq[int],
  opKindsFlat: seq[int],
  leftChildrenFlat: seq[int],
  rightChildrenFlat: seq[int],
  constantsFlat: seq[float64]
): seq[StackProgram] =
  result = newSeq[StackProgram](programSizes.len)
  var offset = 0
  for i, progSize in programSizes:
    var featureIndices = newSeq[int](progSize)
    var opKinds = newSeq[int](progSize)
    var leftChildren = newSeq[int](progSize)
    var rightChildren = newSeq[int](progSize)
    var constants = newSeq[float64](progSize)
    for j in 0..<progSize:
      let idx = offset + j
      featureIndices[j] = featureIndicesFlat[idx]
      opKinds[j] = opKindsFlat[idx]
      leftChildren[j] = leftChildrenFlat[idx]
      rightChildren[j] = rightChildrenFlat[idx]
      constants[j] = constantsFlat[idx]
    result[i] = stackProgramFromSerialized(
      featureIndices, opKinds, leftChildren, rightChildren, constants
    )
    offset += progSize
