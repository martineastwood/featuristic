# FeatureSynthesizer Examples

This directory contains comprehensive examples demonstrating the power of automated feature engineering with FeatureSynthesizer.

## 📚 Examples Overview

### 🚀 Quick Start (Start Here!)
- **[04_quick_start.py](./04_quick_start.py)** - Get started in 5 minutes
  - Complete workflow in under 100 lines
  - **Best for**: First-time users

- **[01_linear_regression_power.py](./01_linear_regression_power.py)** - Demonstrates how feature engineering enables simple linear models to solve complex nonlinear problems
  - **Key insight**: Linear model goes from R²=0.03 to R²=0.68 (+2400%!)
  - **Best for**: Understanding the core value proposition

### 🎯 Feature Selection
- **[02_feature_selection_demo.py](./02_feature_selection_demo.py)** - Using FeatureSelector for evolutionary feature subset selection
  - Demonstrates selecting optimal feature subsets
  - Shows improvement with fewer features
  - Custom objective functions with cross-validation

### 📊 Benchmarks & Tests
- **[03_friedman_benchmark.py](./03_friedman_benchmark.py)** - Classic Friedman #1 benchmark (corrected)
  - Standard test for symbolic regression
  - Challenges tree-based models with `sin(π*x1*x2)` interaction
  - Corrected feature ranges [0,1] for proper baseline

### 🎨 Interpretability
- **[05_interpretability.py](./05_interpretability.py)** - Understanding discovered features
  - 6 methods for interpreting synthesized features
  - Domain knowledge validation
  - Feature importance analysis

### 🎯 Classification
- **[06_classification.py](./06_classification.py)** - Classification with FeatureSynthesizer
  - Auto-detects classification vs regression
  - Creates discriminative features for class separation
  - LogisticRegression + FeatureSynthesizer ≈ Random Forest performance

## 🏃 Running the Examples

All examples can be run directly:

```bash
# Quick start - get started in 5 minutes
python 04_quick_start.py

# See the power - linear regression + feature engineering
python 01_linear_regression_power.py

# Feature selection demo
python 02_feature_selection_demo.py

# Classic benchmark
python 03_friedman_benchmark.py

# Interpretability
python 05_interpretability.py

# Classification
python 06_classification.py
```

## 💡 Key Concepts Demonstrated

### 1. Quick Start
**File**: `04_quick_start.py`

Get started in 5 minutes with a complete workflow showing the basics of automated feature engineering.

### 2. Linear Model + Feature Engineering = Powerful
**File**: `01_linear_regression_power.py`

Linear regression can ONLY model linear relationships. FeatureSynthesizer creates nonlinear features (x², sin(x), x1*x2), allowing linear models to solve complex problems.

**Result**: R² improves from 0.03 to 0.68 (+2400%!)

### 3. Evolutionary Feature Selection
**File**: `02_feature_selection_demo.py`

Using genetic algorithms to automatically find the best feature subset, optimizing custom objectives like cross-validation scores.

### 4. Automated Feature Discovery
**File**: `03_friedman_benchmark.py`

The algorithm discovers mathematical relationships without being told what to look for:
- Multiplicative interactions: `x1 * x2`
- Polynomials: `x³`, `x²`
- Transforms: `sin(x)`, `log(x)`, `exp(x)`
- Compositions: `sin(x1 * x2)`

### 5. Interpretability
**File**: `05_interpretability.py`

Six methods for understanding what the algorithm learned:
- Simple feature inspection
- Individual feature evaluation
- Correlation analysis
- Model-based interpretation
- Visual validation
- Domain knowledge validation

### 6. Classification
**File**: `06_classification.py`

FeatureSynthesizer works for classification problems too:
- Auto-detects classification vs regression
- Optimizes features for accuracy/F1/log loss
- LogisticRegression + feature engineering ≈ Random Forest

## 📈 Expected Performance

| Example | Baseline | Augmented | Improvement |
|---------|----------|-----------|-------------|
| Linear Regression | R²=0.03 | R²=0.68 | +2400% |
| Parabola Fitting | R²=0.01 | R²=0.999 | +9900% |
| Friedman Benchmark | R²=0.85-0.91 | R²=0.92-0.95 | +1-10% |
| Classification | Acc=0.72 | Acc=0.92 | +28% |
| Feature Selection | - | Variable | Depends on dataset |

*Note: Results vary by random seed and parameters*

## 🎓 Learning Path

### Beginner
1. Start with `04_quick_start.py` - Get started in 5 minutes
2. Try `01_linear_regression_power.py` - See the dramatic impact
3. Run `02_feature_selection_demo.py` - Learn feature selection

### Intermediate
4. `03_friedman_benchmark.py` - Standard benchmark comparison
5. `05_interpretability.py` - Understanding discovered features
6. `06_classification.py` - Classification problems

### Advanced
7. Experiment with different parameters and datasets
8. Create custom fitness functions for domain-specific problems
9. Integrate into sklearn pipelines

## 🔧 Common Patterns

### Basic Feature Synthesis
```python
from featuristic import FeatureSynthesizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Synthesize features
synth = FeatureSynthesizer(
    n_features=10,
    population_size=50,
    generations=30,
    fitness="auto",  # Auto-detects mse/r2/log_loss/accuracy
    random_state=42
)

X_train_aug = synth.fit_transform(X_train, y_train)
X_test_aug = synth.transform(X_test)

# Combine original + synthesized
X_train_combined = np.column_stack([X_train, X_train_aug])

# Train model
model = LinearRegression()
model.fit(X_train_combined, y_train)
```

### Feature Selection
```python
from featuristic import FeatureSelector

def objective(X_subset, y):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    model = Ridge(alpha=1.0)
    scores = cross_val_score(model, X_subset, y, cv=3, scoring='neg_mean_squared_error')
    return -scores.mean()  # Minimize

selector = FeatureSelector(
    objective_function=objective,
    population_size=40,
    max_generations=25,
    random_state=42
)

X_selected = selector.fit_transform(X, y)
print(f"Selected {len(selector.selected_features_)} features")
```

## 💬 Notes

- All examples use fixed random seeds for reproducibility
- Evolution can take 30-120 seconds depending on generations
- Some examples require `matplotlib` for plotting
- Most work with minimal dependencies (numpy, pandas, sklearn)

## 🤝 Contributing

Have a great example? Add it here! Make sure to:
1. Include clear comments explaining the problem
2. Show baseline vs. augmented performance
3. Explain WHY the result is interesting
4. Use fixed random seeds for reproducibility

## 📚 Additional Resources

- **Main README**: `../readme.md`
- **API Documentation**: `docs/build/html/`
- **CLAUDE.md**: Developer guide and architecture
