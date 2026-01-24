# Installation

Integrate Featuristic's high-performance feature engineering pipeline into your environment.

## System Requirements

Featuristic is designed to be lightweight and highly compatible. The only hard requirement is a modern Python environment.

* **Python:** Version 3.8 or higher.
* **OS:** Windows, macOS, or Linux.
* **Backend:** Nim is **not** required for installation. The high-performance Nim binaries are pre-compiled and bundled directly with the Python package.

---

## Standard Installation

The recommended way to install Featuristic for production or standard data science workflows is via the Python Package Index (PyPI).

```bash
pip install featuristic

```

This single command installs the core library along with the optimized Nim backend and all required dependencies.

---

## Installation from Source

For advanced users, contributors, or those requiring the bleeding-edge development branch, you can install Featuristic directly from the source repository.

```bash
git clone https://github.com/martineastwood/featuristic.git
cd featuristic
pip install .

```

### Development Mode

If you intend to modify the source code, run the test suite, or contribute to the documentation, install the package in editable mode with the `[dev]` flag. This installs additional tooling including `pytest`, `black`, `pylint`, and `Sphinx`.

```bash
pip install -e .[dev]

```

---

## Verification

To confirm that the package and its underlying Nim binaries are installed correctly, run the following command in your Python terminal:

```python
import featuristic as ft
print(f"Featuristic version: {ft.__version__}")

```

If the version number is returned without errors, your installation is successful and ready for use.

---

## Ecosystem Integration

Featuristic is built to integrate seamlessly with the modern PyData ecosystem. The installer automatically handles the following core dependencies:

* **NumPy (>= 1.25.0) & Pandas (>= 2.0.0):** Used for zero-copy memory management and data manipulation.
* **Scikit-Learn (>= 1.4.0):** Featuristic explicitly inherits from `BaseEstimator` and `TransformerMixin` for total compatibility with standard ML pipelines.
* **Matplotlib (>= 3.0.0):** Required for generating convergence plots and feature history visualizations.
* **Tqdm (>= 4.32.0):** Provides efficient, low-overhead progress tracking during long genetic evolution runs.
* **Ucimlrepo (>= 0.0.5):** Included for easily fetching benchmark datasets during testing and evaluation.
