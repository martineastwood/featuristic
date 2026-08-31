# Release Notes

## v2.0.0

Featuristic 2.0 moves its genetic synthesis, selection, and mRMR hot paths to a
compiled [Nim](https://nim-lang.org/) backend built with
[Nuwa](https://github.com/martineastwood/nuwa-build). The public estimators retain
their familiar scikit-learn-style Python API.

### Highlights

- Supports CPython 3.10–3.14 on Linux, macOS, and Windows.
- Requires `nuwa-build>=0.5.3`, `nimpy@0.2.1`, and `nuwa_sdk@0.4.4` when
  building from source. Binary wheels do not require a local Nim installation.
- Provides native synthesis fitness metrics (`pearson`, `mae`, and `mse`) and
  native feature-selection metrics (`mse`, `mae`, `r2`, `logloss`, and
  `accuracy`).
- Supports custom Python synthesis fitness functions and feature-selection
  objective functions when application-specific scoring is required.
- Accepts an operator subset by name through `functions=` and evaluates it in
  the compiled genetic algorithm.
- Uses deterministic random seeds and supports early termination in both
  estimators.
- Returns up to `n_features` useful synthetic features. If fewer candidates
  survive validation and de-duplication, all available features are returned.
- Reports `2.0.0` from both `featuristic.__version__` and
  `featuristic_lib.getVersion()`.

### Symbolic operators

Synthesis operators are selected by name from Featuristic's compiled operator
catalogue. Arbitrary Python operator callbacks are not part of the synthesis API;
keeping expression evaluation native avoids maintaining separate Python and Nim
execution semantics.
