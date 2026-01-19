# Walkthrough: Standardized Maturin Build Process

I have successfully standardized the build process for `featuristic` using `maturin`. This replaces the previous manual build script with a production-ready workflow that supports local development and automated CI/CD.

## Key Changes

### 1. Build Infrastructure

- **[pyproject.toml](file:///Users/martin/repos/featuristic/pyproject.toml)**: Switched the build backend to `maturin` and configured the Rust manifest path. This allows `pip install .` and `maturin develop` to work seamlessly from the root directory.
- **[release.yml](file:///Users/martin/repos/featuristic/.github/workflows/release.yml)**: Created a new GitHub Actions workflow using `maturin-action`. This will automatically build wheels for:
  - **Linux**: x86_64, aarch64 (using manylinux)
  - **macOS**: Intel and Apple Silicon
  - **Windows**: x64, x86
- The workflow also handles publishing to PyPI automatically when you create a new tag (e.g., `v2.0.0`).

### 2. Hybrid Architecture Compatibility

During verification, I discovered and fixed several issues in the Python code that were blocking the "rustification" process:

- **Module Exports**: Updated `src/featuristic/fitness/__init__.py` to export fitness functions like `mse`, `r2`, etc.
- **Submodule Renaming**: Renamed internal fitness modules (e.g., `mse.py` -> `_mse.py`) to prevent them from shadowing the exported functions. This ensures that `from featuristic.fitness import mse` returns the function as expected.
- **Function Signatures**: Updated fitness functions to support the new hybrid architecture by making `program` and `parsimony` optional and reordering parameters.
- **Numpy Support**: Fixed `is_invalid_prediction` in `src/featuristic/fitness/utils.py` to work with the Numpy arrays returned by the Rust engine.

## Verification Results

The build and integration were verified using the existing test suite. All tests now pass in the new standardized environment.

```bash
source .venv/bin/activate
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
python3 -m pytest tests/integration/test_hybrid_rust_python.py
```

```text
============================= test session starts ==============================
platform darwin -- Python 3.13.1, pytest-9.0.2, pluggy-1.6.0
collected 3 items

tests/integration/test_hybrid_rust_python.py ...                         [100%]

============================== 3 passed in 1.60s ===============================
```

## Best Practice Workflow

### Local Development

To work on the project locally, simply run:

```bash
maturin develop --release
```

This builds the Rust extension and installs it into your current virtual environment in editable mode.

### Releasing to PyPI

1. Update the version in `pyproject.toml`.
2. Commit and push your changes.
3. Create and push a new tag:
   ```bash
   git tag v2.0.0
   git push origin v2.0.0
   ```
4. GitHub Actions will handle the rest!
