Release Notes
---------------

v1.1.0 April 10, 2024
========================

- **Changes:**
    - The `functions` parameter in the `GeneticFeatureSynthesis` class now accepts a list of strings representing the names of the functions to be used in the genetic programming process. The default value is `['add', 'sub', 'mul', 'div', 'square', 'cube', 'abs' 'negate', 'sin', 'cos', 'tan']`. The full list of built in functions can be found in the `list_operations` function.
    - Added `SymbolicMulConstant` and `SymbolicAddConstant` symbolic functions. These can be useful where their is an offset to the data but are not currently used by default as there is a risk of overfitting where an offset is not present.
    - Renamed `list_operations` to `list_symbolic_functions` for consistency
    - Added `CustomSymbolicFunction` class to allow users to define their own symbolic functions to be used in the genetic feature synthesis process.
    - Updated unit tests to reflect changes
- **Documentation:**
    - Added example showing use of custom symbolic functions.


v1.0.1 April 4, 2024
====================

- **Changes:**
    - Added `tournament_size` parameter to GeneticFeatureSelection class and set default to 10
    - Set default tournament size to 10 for GeneticFeatureSynthesis class
- **Documentation:**
    - Updated README.md example
    - Updated example notebooks
    - Added eplanation of tournament selection in the `Tuning the Genetic Feature Synthesis` guide


v1.0.0 Mar 30, 2024
====================

- Initial release
