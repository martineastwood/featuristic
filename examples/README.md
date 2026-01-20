# Featuristic Example Notebook

This example demonstrates the complete featuristic pipeline with the Nim backend.

## Quick Start

```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import featuristic as ft

# 1. Create dataset
X_raw, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=7,
    n_redundant=4,
    random_state=42
)
X = pd.DataFrame(X_raw, columns=[f"f_{i}" for i in range(20)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Generate new features with genetic programming (Nim backend)
synth = ft.GeneticFeatureSynthesis(
    num_features=25,
    population_size=100,
    max_generations=25,
    parsimony_coefficient=0.001
)

# 3. Fit and transform (zero-copy mRMR selection included!)
X_train_new = synth.fit_transform(X_train, y_train)
X_test_new = synth.transform(X_test)

# 4. Train and evaluate
model = LogisticRegression(max_iter=1000)
model.fit(X_train_new, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test_new))

print(f"Accuracy with engineered features: {accuracy:.4f}")
```

## Performance

- **Genetic Algorithm**: 10-50x faster (complete evolution in Nim)
- **mRMR Selection**: 38x faster (Nim implementation)
- **Memory**: Zero-copy numpy arrays for both features and target

## API Reference

### GeneticFeatureSynthesis Parameters

- `num_features` (int): Number of features to generate (default: 10)
- `population_size` (int): GA population size (default: 100)
- `max_generations` (int): Number of GA generations (default: 25)
- `tournament_size` (int): Tournament selection size (default: 10)
- `crossover_proba` (float): Crossover probability (default: 0.85)
- `parsimony_coefficient` (float): Complexity penalty (default: 0.001)
- `early_termination_iters` (int): Early stopping patience (default: 15)
- `pbar` (bool): Show progress bar (default: True)
- `n_jobs` (int): Parallel jobs (default: -1, use 1 for Nim speedup)

### Key Methods

- `fit(X, y)`: Fit the feature generator
- `transform(X)`: Transform features
- `fit_transform(X, y)`: Fit and transform
- `get_feature_info()`: Get generated feature formulas
- `plot_history()`: Plot GA convergence

## Running the Example

```bash
# Run the Python script
python examples/featuristic_example.py
```

## Notes

- The pipeline uses the Nim backend automatically for maximum performance
- mRMR feature selection happens automatically in `fit_transform`
- Generated features are retained and selected based on mRMR scores
- Use `random_state` parameter for reproducible results
