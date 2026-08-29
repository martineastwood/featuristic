# Release Notes

## v2.0.0 — unreleased (Nim branch)

Compiled backend via [Nuwa](https://github.com/martineastwood/nuwa-build). **Not on PyPI.** Do not tag this as a public 2.0 until wheel CI is green and install docs match.

### Planned / in this branch

- Genetic synthesis, selection, and mRMR hot path in Nim; sklearn API in Python.
- CPython 3.10–3.14. Python 3.8 and 3.9 are not supported on this line.
- Pins: `nuwa-build>=0.5.1`, `nimpy@0.2.1`, `nuwa_sdk@0.4.4`.
- Version string is `2.0.0` in Python and in `featuristic_lib.getVersion()`.
- Parallel GA uses `std/typedthreads`, not weave.
- Mutation node pick is uniform; 1.1 used depth-weighted selection (intentional 2.0 difference until revisited).

### Not done before go-live

- PyPI wheels / Trusted Publishing
- Deploying this docs tree to featuristic.co.uk
- Merging `nim` over `main` (would replace 1.1.0)

## v1.1.0 - April 10, 2024

### Changes
- The `functions` parameter in the `GeneticFeatureSynthesis` class now accepts a list of strings representing the names of the functions to be used in the genetic programming process. The default value is `['add', 'sub', 'mul', 'div', 'square', 'cube', 'abs' 'negate', 'sin', 'cos', 'tan']`. The full list of built in functions can be found in the `list_symbolic_functions` function.
- Added `SymbolicMulConstant` and `SymbolicAddConstant` symbolic functions. These can be useful where their is an offset to the data but are not currently used by default as there is a risk of overfitting where an offset is not present.
- Renamed `list_operations` to `list_symbolic_functions` for consistency
- Added `CustomSymbolicFunction` class to allow users to define their own symbolic functions to be used in the genetic feature synthesis process.
- Updated unit tests to reflect changes

### Documentation
- Added example showing use of custom symbolic functions.

## v1.0.1 - April 4, 2024

### Changes
- Added `tournament_size` parameter to GeneticFeatureSelection class and set default to 10
- Set default tournament size to 10 for GeneticFeatureSynthesis class

### Documentation
- Updated README.md example
- Updated example notebooks
- Added explanation of tournament selection in the `Tuning the Genetic Feature Synthesis` guide

## v1.0.0 - March 30, 2024

### Initial Release
