<p align="center">
<img width=50% src="https://raw.githubusercontent.com/martineastwood/featuristic/dev/docs/_static/logo.png" alt="Featuristic" />
</p>

<p align="center">
<i>"Because feature engineering should be a science, not an art."</i>
</p>

<div align="center">

<a href="">[![Python Version](https://img.shields.io/pypi/pyversions/featuristic)](https://pypi.org/project/featuristic/)</a>
<a href="">[![PyPI](https://img.shields.io/pypi/v/featuristic.svg)](https://pypi.org/project/featuristic/)</a>
<a href="">[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)</a>
<a href='https://coveralls.io/github/martineastwood/featuristic?branch=dev'><img src='https://coveralls.io/repos/github/martineastwood/featuristic/badge.svg?branch=dev' alt='Coverage Status' /></a>
<a href="">[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)</a>
<a href="">[![Code style: pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)</a>

</div>

# 🧬 Featuristic

**Breeding Smarter Features.**

Featuristic is a high-performance automated feature engineering library powered by **Rust**. It evolved from search for interpretable, high-performance features using symbolic regression and genetic programming.

By offloading heavy computations to Rust, Featuristic is **5-20x faster** than traditional Python-based symbolic regression tools while maintaining a seamless Scikit-learn compatible API.

## 🚀 Key Features

- **Blazing Fast**: Core evolutionary engine implemented in Rust.
- **Symbolic Programs**: Discover mathematical formulas that capture non-linear relationships.
- **Interpretable**: Generated features are human-readable mathematical expressions.
- **Parsimony-Aware**: Built-in penalties for overly complex "bloated" expressions.
- **Smart Selection**: Uses **Maximum Relevance Minimum Redundancy (mRMR)** to pick the best features.
- **Scikit-learn Compatible**: Works perfectly with `Pipeline`, `GridSearchCV`, and `cross_val_score`.

## 🔧 Installation

```bash
pip install featuristic
```

Or from source (requires Rust):

```bash
git clone https://github.com/martineastwood/featuristic.git
cd featuristic
pip install -e .
```

## 🧪 Quickstart: Automated Feature Synthesis

Discover new features from your data in just a few lines of code.

```python
import pandas as pd
from sklearn.datasets import fetch_california_housing
from featuristic import FeatureSynthesizer

# Load some data
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Synthesize 5 new features
synth = FeatureSynthesizer(n_features=5, generations=20)
X_new = synth.fit_transform(X, y)

# Inspect the evolved formulas
for p in synth.get_programs():
    print(f"Feature: {p['expression']}")
```

## 🧩 Evolutionary Feature Selection

Optimize your feature subset using a genetic algorithm.

```python
from featuristic import FeatureSelector
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Define a custom objective function (lower is better)
def objective(X_subset, y):
    model = Ridge().fit(X_subset, y)
    return mean_squared_error(y, model.predict(X_subset))

# Select the best feature subset
selector = FeatureSelector(objective_function=objective, max_generations=50)
X_selected = selector.fit_transform(X, y)

print(f"Selected features: {selector.selected_features_}")
```

## 🔌 Seamless Integration with Scikit-learn

Featuristic components can be dropped into any Scikit-learn pipeline.

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from featuristic import FeatureSynthesizer, FeatureSelector

# Build a robust ML pipeline
pipeline = Pipeline([
    ("synth", FeatureSynthesizer(n_features=10, generations=20)),
    ("select", FeatureSelector(objective_function=objective, max_generations=20)),
    ("model", GradientBoostingRegressor())
])

pipeline.fit(X, y)
```

## 📚 Documentation

[👉 Read the full docs](https://www.featuristic.co.uk/)

## 🧠 Why Use Featuristic?

- ✅ **Performance**: Rust-powered engine handles large populations and many generations with ease.
- ✅ **Interpretability**: No "black-box" features. You get actual formulas you can understand and trust.
- ✅ **Automation**: Finds non-linear transformations and interactions without manual trial-and-error.

## 🛠️ Contributing

Pull requests are welcome!

## 📄 License

MIT
