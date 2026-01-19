# Core data types for featuristic genetic programming
# This module defines the fundamental types used for symbolic regression

type
  ## Enumeration of all symbolic operations
  OperationKind* = enum
    opAdd = "add"              # Addition: a + b
    opSubtract = "subtract"    # Subtraction: a - b
    opMultiply = "multiply"    # Multiplication: a * b
    opDivide = "divide"        # Division: a / b (safe)
    opAbs = "abs"              # Absolute value: |a|
    opNegate = "negate"        # Negation: -a
    opSin = "sin"              # Sine: sin(a)
    opCos = "cos"              # Cosine: cos(a)
    opTan = "tan"              # Tangent: tan(a)
    opSqrt = "sqrt"            # Square root: sqrt(|a|)
    opSquare = "square"        # Square: a²
    opCube = "cube"            # Cube: a³
    opAddConstant = "add_constant"   # Add constant: a + c
    opMulConstant = "mul_constant"   # Multiply constant: a * c
    opFeature = "feature"      # Feature leaf node (for stack-based evaluation)

  ## A node in the program tree
  ## Can be either an operation (internal node) or a feature (leaf node)
  Node* = ref object
    case kind*: OperationKind
    of opAddConstant:
      addConstant*: float64  # Constant value to add
    of opMulConstant:
      mulConstant*: float64  # Constant value to multiply
    else:
      discard  # No additional fields for other operations

    children*: seq[Node]     # Child nodes (empty for leaf nodes)
    featureName*: string     # Feature name (only for leaf nodes)

  ## A program is a tree of nodes (distinct type for type safety)
  Program* = distinct Node

  ## Collection of programs representing a population
  ProgramList* = seq[Program]

  ## Population of programs for genetic programming
  PopulationObj* = object
    programs*: ProgramList       # The programs in the population
    populationSize*: int         # Number of programs
    operations*: seq[OperationKind]  # Available operations
    tournamentSize*: int         # Size of tournament for selection
    crossoverProb*: float64      # Probability of crossover vs mutation

  Population* = ref PopulationObj

# Export the types for use in Python
export OperationKind, Node, Program, ProgramList, Population
