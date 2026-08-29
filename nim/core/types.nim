# Symbolic operations for stack programs (single source of truth for Python metadata)

type
  OperationKind* = enum
    opAdd = "add"
    opSubtract = "subtract"
    opMultiply = "multiply"
    opDivide = "divide"
    opAbs = "abs"
    opNegate = "negate"
    opSin = "sin"
    opCos = "cos"
    opTan = "tan"
    opSqrt = "sqrt"
    opSquare = "square"
    opCube = "cube"
    opPow = "pow"
    opAddConstant = "add_constant"
    opMulConstant = "mul_constant"
    opFeature = "feature"
