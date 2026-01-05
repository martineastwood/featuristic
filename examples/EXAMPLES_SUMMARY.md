# Examples Summary

This document provides a quick overview of all examples in the `examples/` directory.

## Available Examples

### 1. Quick Start (04_quick_start.py)
**Purpose**: Get started in 5 minutes

**What it demonstrates**:
- Complete workflow in under 100 lines
- Basic FeatureSynthesizer usage
- Model training and evaluation
- Feature importance analysis

**Best for**: First-time users who want to see the basics quickly

**Key Results**:
- Shows baseline vs augmented performance
- Demonstrates feature importance
- Typical runtime: 30-60 seconds

---

### 2. Linear Regression Power (01_linear_regression_power.py)
**Purpose**: Demonstrate that feature engineering enables simple models to solve complex problems

**What it demonstrates**:
- Linear models fail on nonlinear problems without feature engineering
- FeatureSynthesizer creates x², sin(x), x₁*x₂ features
- Linear model + FeatureSynthesizer ≈ Random Forest performance

**Best for**: Understanding the core value proposition

**Key Results**:
- Test Case 1: R² improves from 0.03 to 0.68 (+2400%!)
- Test Case 2: R² improves from 0.01 to 0.999 (perfect parabola fit)
- Achieves ~90% of Random Forest performance with interpretable model

---

### 3. Feature Selection Demo (02_feature_selection_demo.py)
**Purpose**: Demonstrate evolutionary feature subset selection

**What it demonstrates**:
- FeatureSelector with custom objective functions
- Cross-validation objectives
- Handling correlated features
- Complexity penalty objectives

**Best for**: Learning when to use feature selection vs feature synthesis

**Key Results**:
- Reduces 100 features to ~10-15 most important
- Maintains or improves performance with fewer features
- Demonstrates custom objective functions

---

### 4. Friedman Benchmark (03_friedman_benchmark.py)
**Purpose**: Classic symbolic regression benchmark

**What it demonstrates**:
- Standard test for symbolic regression algorithms
- Challenges tree-based models with sin(π*x₁*x₂) interaction
- Corrected feature ranges [0,1] (bug in original benchmark)

**Best for**: Comparing against standard benchmarks

**Key Results**:
- Baseline (GradientBoosting): R² ≈ 0.85-0.91
- With FeatureSynthesizer: R² ≈ 0.92-0.95
- Discovers sin interaction and polynomial features

**Why this matters**: Friedman #1 is a standard test in symbolic regression literature

---

### 5. Interpretability (05_interpretability.py)
**Purpose**: Understanding and validating discovered features

**What it demonstrates**:
- 6 methods for interpreting synthesized features:
  1. Simple feature inspection
  2. Individual feature evaluation
  3. Correlation analysis
  4. Model-based interpretation
  5. Visual validation
  6. Domain knowledge validation

**Best for**: Learning how to understand what the algorithm discovered

**Key Results**:
- Shows how to examine feature expressions
- Validates features match domain knowledge
- Demonstrates feature importance analysis

---

### 6. Classification (06_classification.py)
**Purpose**: Demonstrate FeatureSynthesizer works for classification

**What it demonstrates**:
- Auto-detects classification vs regression
- Creates features optimized for accuracy/F1/log loss
- LogisticRegression + FeatureSynthesizer ≈ Random Forest

**Best for**: Understanding classification capabilities

**Key Results**:
- Baseline (LogisticRegression): 67.5% accuracy
- With FeatureSynthesizer: 69.5% accuracy (+3%)
- Demonstrates discriminative feature creation

---

## Statistics

- **Total Examples**: 6
- **Total Lines of Code**: 1,781
- **Average Lines per Example**: ~300
- **Languages Used**: Python
- **Dependencies**: numpy, pandas, sklearn, featuristic

## When to Use Each Example

### I'm new to FeatureSynthesizer...
Start with: `04_quick_start.py`
Then try: `01_linear_regression_power.py`

### I want to understand feature selection...
Run: `02_feature_selection_demo.py`

### I want to compare against benchmarks...
Run: `03_friedman_benchmark.py`

### I need to interpret my results...
Run: `05_interpretability.py`

### I have a classification problem...
Run: `06_classification.py`

### I want to see the power of automated feature engineering...
Run: `01_linear_regression_power.py`

## Common Patterns Across Examples

### 1. Always use a random seed
```python
synth = FeatureSynthesizer(
    random_state=42,
    ...
)
```

### 2. Combine original + synthesized features
```python
X_train_combined = np.column_stack([X_train, X_train_aug])
```

### 3. Use parsimony_coefficient to control complexity
```python
synth = FeatureSynthesizer(
    parsimony_coefficient=0.005,  # Higher = simpler features
    ...
)
```

### 4. Inspect discovered features
```python
programs = synth.get_programs()
for prog in programs:
    print(prog['expression'])
```

## Key Takeaways

1. **FeatureSynthesizer works for both regression and classification**
2. **Simple models + good features ≈ complex models**
3. **Features are interpretable (unlike neural networks)**
4. **Auto-detection of problem type (regression vs classification)**
5. **Reproducible results with random_state**

## Performance Summary

| Example | Metric | Baseline | Augmented | Improvement |
|---------|--------|----------|-----------|-------------|
| Linear Regression | R² | 0.03 | 0.68 | +2400% |
| Parabola | R² | 0.01 | 0.999 | +9900% |
| Friedman | R² | 0.85-0.91 | 0.92-0.95 | +1-10% |
| Classification | Accuracy | 67.5% | 69.5% | +3% |

*Note: Results vary by random seed and parameters*

## Next Steps

After exploring these examples:

1. **Try your own dataset**: Replace synthetic data with your real data
2. **Experiment with parameters**:
   - `n_features`: Try 5, 10, 20, 50
   - `generations`: Try 25, 50, 100
   - `parsimony_coefficient`: Try 0.001, 0.005, 0.01
   - `fitness`: Try "mse", "r2", "accuracy", "log_loss"
3. **Integrate into pipelines**: Use with sklearn.pipeline.Pipeline
4. **Create custom fitness functions**: See `fitness/registry.py`

## Getting Help

- **Documentation**: See `readme.md` and `docs/`
- **Architecture**: See `CLAUDE.md`
- **Issues**: Report bugs at GitHub issues
- **Examples**: This directory!
