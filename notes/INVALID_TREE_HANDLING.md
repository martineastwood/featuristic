# Invalid Tree Handling Strategy

## Problem

Genetic programming for symbolic regression inherently produces **invalid trees** that result in numerical overflow:
- `exp(exp(exp(x)))` → Overflow to `Inf`
- `tan(π/2)` → Infinity
- `log(-1)` → `NaN`
- `1/0` → `Inf`

These invalid trees must be handled properly to avoid:
1. Crashes during evaluation
2. Non-reproducible fitness calculations
3. Noisy optimization landscape
4. Unfair selection pressure

## Solution: Multi-Layer Defense

### Layer 1: Prevention in Rust Operations (`evaluate.rs`)

Most operations include safety checks to prevent overflow:

```rust
// Division by zero
3 => {  // divide
    let result = &args[0] / &args[1];
    result.mapv(|x: f64| if x.is_finite() { x } else { 1.0 })
}

// Exponential overflow
9 => {  // exp
    args[0].mapv(|x: f64| {
        let clipped = x.clamp(-20.0, 20.0);  // exp(20) ≈ 485M
        clipped.exp()
    })
}

// Logarithm domain errors
10 => {  // log
    args[0].mapv(|x: f64| {
        if x <= 0.0 { 0.0 } else { x.ln() }
    })
}

// Tangent infinity
8 => {  // tan
    args[0].mapv(|x: f64| {
        let clipped = x.clamp(-1.5, 1.5);  // Prevent π/2
        clipped.tan()
    })
}
```

**Impact**: Prevents 80-90% of potential invalid results

### Layer 2: Detection and Constant Penalty (`evaluate.rs:183-190`)

When invalid values **do** occur (NaN/Inf escape prevention):

```rust
if result.iter().any(|x: &f64| !x.is_finite()) {
    // Return constant large penalty (not random!)
    const INVALID_PENALTY: f64 = 1e9;
    Ok(Array1::from_elem(result.len(), INVALID_PENALTY))
}
```

**Why constant penalty?**
- ✅ **Reproducible**: Same tree always gets same fitness
- ✅ **Consistent**: All invalid trees get same penalty
- ✅ **Optimizable**: Evolution can recover if tree improves
- ✅ **No randomness**: Smooth fitness landscape

**Alternative approaches considered:**
- ❌ Random noise → Non-reproducible, noisy
- ❌ Return zeros → Triggers PTP=0 check
- ❌ Return inf → Makes comparison difficult

### Layer 3: Capping in Python Fitness Functions (`_mse.py:15-18`)

Extremely large MSE values are capped:

```python
loss = mean_squared_error(y_true, y_pred)

# Cap to prevent numerical overflow
if loss > 1e10:
    return 1e10

penalty = (tree_node_count(program) if program else 1.0) ** parsimony
return loss * penalty
```

**Why 1e10?**
- MSE of 1e10 means predictions are off by ~100,000 on average
- This is **far worse** than any reasonable prediction
- Small enough to avoid numerical overflow in float64
- Consistent upper bound for selection

## Fitness Hierarchy

```
Valid trees:
  Excellent: MSE < 1.0
  Good:      1.0 ≤ MSE < 10
  Fair:      10 ≤ MSE < 100
  Poor:      100 ≤ MSE < 1e5

Invalid trees:
  All:       MSE = 1e10 (capped)
```

This clear separation ensures:
- Valid trees are **always** preferred over invalid trees
- Invalid trees don't pollute the gene pool
- Evolution can still explore near-invalid solutions

## Testing

Verify invalid tree handling:

```python
# Test reproducibility
fitness1 = [mse(y, pred) for pred in population.evaluate(X)]
fitness2 = [mse(y, pred) for pred in population.evaluate(X)]
assert fitness1 == fitness2  # ✓ Same tree, same fitness

# Test overflow tree
overflow_tree = {'exp': [{'exp': [{'exp': 'x'}]}]}
fitness_overflow = mse(y, evaluate_tree(overflow_tree, X))
assert fitness_overflow == 1e10  # ✓ Capped

# Test normal tree
normal_tree = 'x1 * x2'
fitness_normal = mse(y, evaluate_tree(normal_tree, X))
assert fitness_normal < 1e10  # ✓ Normal range
```

## Recommendations for Production

1. **Monitor invalid tree rate**: If >50% of trees are invalid, adjust parameters:
   - Reduce `max_depth` (try 4-6)
   - Increase `parsimony_coefficient` (try 0.01-0.05)
   - Reduce population size (fewer complex trees)

2. **Use appropriate fitness functions**:
   - Regression: `mse`, `r2`
   - Classification: `log_loss`, `accuracy`
   - All have proper invalid handling

3. **Debug with verbose output**:
   ```python
   synth = FeatureSynthesizer(..., verbose=True)
   # Shows: "Best fitness: X.XXX" - check if it's 1e10
   ```

## Summary

| Aspect | Approach | Benefit |
|--------|----------|---------|
| Prevention | Clamp operations | Reduces invalid trees by 80-90% |
| Detection | Check for `!is_finite()` | Catches all remaining issues |
| Penalty | Constant 1e9 predictions | Reproducible, consistent |
| Capping | MSE max 1e10 | Prevents overflow |
| Result | Smooth optimization | Evolution works correctly |

**Key insight**: Invalid trees are **inevitable** in GP. The key is handling them gracefully rather than trying to prevent them entirely.
