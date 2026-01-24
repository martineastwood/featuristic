# Symbolic operations for genetic programming
# Mathematical functions used in program trees with vectorized NumPy support
# Note: This file is included in featuristic_lib.nim, not compiled separately
# All nimpy exports are handled by the main module

import std/math

# ============================================================================
# Vectorized Operations using zero-copy NumPy array access
# ============================================================================

## Vectorized safe division for NumPy arrays (zero-copy)
proc safeDivVecImpl(ptrA: int, ptrB: int, length: int): seq[float64] =
  ## Zero-copy vectorized safe division
  ## Input: pointers to two float64 arrays
  ## Returns: new array with a/b where b != 0, else a
  let dataA = cast[ptr UncheckedArray[float64]](ptrA)
  let dataB = cast[ptr UncheckedArray[float64]](ptrB)
  result = newSeq[float64](length)
  for i in 0..<length:
    if abs(dataB[i]) < 1e-10:
      result[i] = dataA[i]
    else:
      result[i] = dataA[i] / dataB[i]

## Vectorized negate for NumPy arrays (zero-copy)
proc negateVecImpl(ptrA: int, length: int): seq[float64] =
  let dataA = cast[ptr UncheckedArray[float64]](ptrA)
  result = newSeq[float64](length)
  for i in 0..<length:
    result[i] = -dataA[i]

## Vectorized square for NumPy arrays (zero-copy)
proc squareVecImpl(ptrA: int, length: int): seq[float64] =
  let dataA = cast[ptr UncheckedArray[float64]](ptrA)
  result = newSeq[float64](length)
  for i in 0..<length:
    result[i] = dataA[i] * dataA[i]

## Vectorized cube for NumPy arrays (zero-copy)
proc cubeVecImpl(ptrA: int, length: int): seq[float64] =
  let dataA = cast[ptr UncheckedArray[float64]](ptrA)
  result = newSeq[float64](length)
  for i in 0..<length:
    result[i] = dataA[i] * dataA[i] * dataA[i]

## Vectorized sin for NumPy arrays (zero-copy)
proc sinVecImpl(ptrA: int, length: int): seq[float64] =
  let dataA = cast[ptr UncheckedArray[float64]](ptrA)
  result = newSeq[float64](length)
  for i in 0..<length:
    result[i] = sin(dataA[i])

## Vectorized cos for NumPy arrays (zero-copy)
proc cosVecImpl(ptrA: int, length: int): seq[float64] =
  let dataA = cast[ptr UncheckedArray[float64]](ptrA)
  result = newSeq[float64](length)
  for i in 0..<length:
    result[i] = cos(dataA[i])

## Vectorized tan for NumPy arrays (zero-copy)
proc tanVecImpl(ptrA: int, length: int): seq[float64] =
  let dataA = cast[ptr UncheckedArray[float64]](ptrA)
  result = newSeq[float64](length)
  for i in 0..<length:
    result[i] = tan(dataA[i])

## Vectorized power for NumPy arrays (zero-copy, binary)
proc powVecImpl(ptrA: int, ptrB: int, length: int): seq[float64] =
  let dataA = cast[ptr UncheckedArray[float64]](ptrA)
  let dataB = cast[ptr UncheckedArray[float64]](ptrB)
  result = newSeq[float64](length)
  for i in 0..<length:
    let base = dataA[i]
    let exp = dataB[i]
    # Handle special cases for safety
    if abs(base) < 1e-10 and exp < 0:
      # 0^(-n) would be infinity, return 1 instead
      result[i] = 1.0
    elif base < 0 and floor(exp) != exp:
      # Negative base with non-integer exponent would be complex
      # Use absolute value instead
      result[i] = pow(abs(base), exp)
    else:
      result[i] = pow(base, exp)

## Vectorized sqrt for NumPy arrays (zero-copy)
proc sqrtVecImpl(ptrA: int, length: int): seq[float64] =
  let dataA = cast[ptr UncheckedArray[float64]](ptrA)
  result = newSeq[float64](length)
  for i in 0..<length:
    result[i] = sqrt(abs(dataA[i]))

## Vectorized abs for NumPy arrays (zero-copy)
proc absVecImpl(ptrA: int, length: int): seq[float64] =
  let dataA = cast[ptr UncheckedArray[float64]](ptrA)
  result = newSeq[float64](length)
  for i in 0..<length:
    result[i] = abs(dataA[i])

## Vectorized add for NumPy arrays (zero-copy)
proc addVecImpl(ptrA: int, ptrB: int, length: int): seq[float64] =
  let dataA = cast[ptr UncheckedArray[float64]](ptrA)
  let dataB = cast[ptr UncheckedArray[float64]](ptrB)
  result = newSeq[float64](length)
  for i in 0..<length:
    result[i] = dataA[i] + dataB[i]

## Vectorized subtract for NumPy arrays (zero-copy)
proc subVecImpl(ptrA: int, ptrB: int, length: int): seq[float64] =
  let dataA = cast[ptr UncheckedArray[float64]](ptrA)
  let dataB = cast[ptr UncheckedArray[float64]](ptrB)
  result = newSeq[float64](length)
  for i in 0..<length:
    result[i] = dataA[i] - dataB[i]

## Vectorized multiply for NumPy arrays (zero-copy)
proc mulVecImpl(ptrA: int, ptrB: int, length: int): seq[float64] =
  let dataA = cast[ptr UncheckedArray[float64]](ptrA)
  let dataB = cast[ptr UncheckedArray[float64]](ptrB)
  result = newSeq[float64](length)
  for i in 0..<length:
    result[i] = dataA[i] * dataB[i]

## Add constant (for constant operations) - zero-copy
proc addConstantVecImpl(ptrA: int, length: int, constant: float64): seq[float64] =
  let dataA = cast[ptr UncheckedArray[float64]](ptrA)
  result = newSeq[float64](length)
  for i in 0..<length:
    result[i] = dataA[i] + constant

## Multiply constant - zero-copy
proc mulConstantVecImpl(ptrA: int, length: int, constant: float64): seq[float64] =
  let dataA = cast[ptr UncheckedArray[float64]](ptrA)
  result = newSeq[float64](length)
  for i in 0..<length:
    result[i] = dataA[i] * constant

# ============================================================================
# Core Operations (scalar)
# ============================================================================

## Safe division - avoids division by zero
## Returns a if b is close to zero, otherwise a / b
proc safeDiv*(a: float64, b: float64): float64 {.inline.} =
  if abs(b) < 1e-10:
    return a
  return a / b

## Negation - multiply by -1
proc negate*(a: float64): float64 {.inline.} =
  return -a

## Square operation
proc square*(a: float64): float64 {.inline.} =
  return a * a

## Cube operation
proc cube*(a: float64): float64 {.inline.} =
  return a * a * a

## Sine operation
proc sinOp*(a: float64): float64 {.inline.} =
  return sin(a)

## Cosine operation
proc cosOp*(a: float64): float64 {.inline.} =
  return cos(a)

## Tangent operation
proc tanOp*(a: float64): float64 {.inline.} =
  return tan(a)

## Power operation (safe)
proc powOp*(a: float64, b: float64): float64 {.inline.} =
  ## Safe power operation that handles edge cases
  if abs(a) < 1e-10 and b < 0:
    # 0^(-n) would be infinity, return 1 instead
    return 1.0
  elif a < 0 and floor(b) != b:
    # Negative base with non-integer exponent would be complex
    # Use absolute value instead
    return pow(abs(a), b)
  else:
    return pow(a, b)

## Square root operation (of absolute value)
proc sqrtOp*(a: float64): float64 {.inline.} =
  return sqrt(abs(a))

## Absolute value
proc absOp*(a: float64): float64 {.inline.} =
  return abs(a)

# Export scalar operations for internal use
export safeDiv, negate, square, cube, sinOp, cosOp, tanOp, powOp, sqrtOp, absOp
