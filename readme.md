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

An evolutionary feature engineering library based on symbolic regression and genetic programming for interpretable, high-performance models.


## 🚀 What is Featuristic?

Featuristic is an automated feature engineering tool powered by **Evolutionary Feature Synthesis (EFS)**. It evolves symbolic mathematical expressions to discover **high-quality, interpretable features** from your raw data.

- ✅ Symbolic programs
- ✅ Genetic programming
- ✅ Parsimony-aware fitness
- ✅ Maximum Relevance Minimum Redundancy (mRMR)
- ✅ Scikit-learn compatible


## 🔧 Installation

```bash
pip install featuristic
```

Or from source:

```bash
git clone https://github.com/martineastwood/featuristic.git
cd featuristic
pip install -e .
```

## 🧪 Quickstart

```python
from featuristic import FeatureSynthesis
from featuristic.datasets import fetch_wine_dataset

X, y = fetch_wine_dataset()

efs = FeatureSynthesis(num_features=5, max_generations=30)
X_new = efs.fit_transform(X, y)

efs.get_feature_info()
efs.plot_history()
```

## 🧩 Also Included

### `FeatureSelector`

Evolutionary feature **subset** selection using binary genome optimization.

```python
from featuristic import FeatureSelector

fs = FeatureSelector(objective_function=my_cost_fn)
X_selected = fs.fit_transform(X, y)
```

## 🔌 Works Seamlessly With Scikit-learn:

- `Pipeline`, `GridSearchCV`, `cross_val_score`
- scikit-learn models (RandomForest, XGBoost, etc.)
- Custom fitness functions and symbolic ops


```python
from featuristic import FeatureSynthesis, FeatureSelector
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss

# Define a selector objective
def objective(X_subset, y):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=500).fit(X_subset, y)
    probs = clf.predict_proba(X_subset)
    return log_loss(y, probs)

pipeline = Pipeline([
    ("synthesis", FeatureSynthesis(num_features=20, max_generations=25)),
    ("select", FeatureSelector(objective_function=objective, max_generations=30)),
    ("model", GradientBoostingClassifier())
])

pipeline.fit(X, y)
```


## 📚 Documentation

[👉 Read the full docs](https://www.featuristic.co.uk/)

- Evolutionary feature synthesis
- Evolutionary feature selection
- Symbolic functions & primitives
- Fitness customization
- Use in sklearn pipelines


## 🧠 Why Use Featuristic?

- ✅ Produces human-readable feature formulas
- ✅ Supports classification & regression
- ✅ Requires no domain-specific heuristics

## 🛠️ Contributing

Pull requests welcome!

## 📄 License

MIT
