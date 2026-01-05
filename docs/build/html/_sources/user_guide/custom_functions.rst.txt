Custom Symbolic Functions
========================

.. contents:: Table of Contents
   :local:
   :depth: 2

Introduction
------------

FeatureSynthesizer includes a comprehensive set of built-in symbolic functions (add, subtract, multiply, divide, sin, cos, log, sqrt, etc.). However, you can extend this set with **domain-specific functions** for your use case.

**Why Custom Functions?**

* Domain-specific operations (e.g., financial formulas)
* Problem-specific transformations (e.g., signal processing)
* Performance optimization (specialized calculations)
* Experimental new operations

**Important:** Custom functions currently require **modifying Rust code** and rebuilding the extension.

Current Limitations
--------------------

**As of this version:**

* ✅ Can use **built-in functions** by name
* ✅ Can **combine built-in functions** in expressions
* ❌ Cannot add custom functions from Python
* ❌ Custom functions require **Rust implementation**

**Future plans:**

* Python-side custom function registration
* Plugin system for dynamic function loading
* User-contributed function library

Built-in Functions Reference
-----------------------------

Arithmetic Operations
^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Built-in Arithmetic Functions
   :widths: 20 30 50
   :header-rows: 1

   * - Name
     - Symbol
     - Description

   * - add
     - ``+``
     - Addition: ``a + b``

   * - sub
     - ``-``
     - Subtraction: ``a - b``

   * - mul
     - ``*``
     - Multiplication: ``a * b``

   * - div
     - ``/``
     - Division: ``a / b`` (protected from division by zero)

   * - neg
     - ``-x``
     - Negation: ``-x``

Trigonometric Functions
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Built-in Trigonometric Functions
   :widths: 20 30 50
   :header-rows: 1

   * - Name
     - Symbol
     - Description

   * - sin
     - ``sin(x)``
     - Sine: ``sin(x)``

   * - cos
     - ``cos(x)``
     - Cosine: ``cos(x)``

   * - tan
     - ``tan(x)``
     - Tangent: ``tan(x)``

Exponential and Logarithmic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Built-in Exponential Functions
   :widths: 20 30 50
   :header-rows: 1

   * - Name
     - Symbol
     - Description

   * - exp
     - ``exp(x)``
     - Exponential: ``e^x``

   * - log
     - ``log(x)``
     - Natural log: ``ln(x)`` (protected from log(0))

   * - sqrt
     - ``sqrt(x)``
     - Square root: ``√x`` (protected from sqrt(negative))

Power Functions
^^^^^^^^^^^^^^

.. list-table:: Built-in Power Functions
   :widths: 20 30 50
   :header-rows: 1

   * - Name
     - Symbol
     - Description

   * - square
     - ``x²``
     - Square: ``x^2``

   * - cube
     - ``x³``
     - Cube: ``x^3``

Other Functions
^^^^^^^^^^^^^^

.. list-table:: Other Built-in Functions
   :widths: 20 30 50
   :header-rows: 1

   * - Name
     - Symbol
     - Description

   * - abs
     - ``|x|``
     - Absolute value: ``|x|``

   * - min
     - ``min(a, b)``
     - Minimum: ``min(a, b)``

   * - max
     - ``max(a, b)``
     - Maximum: ``max(a, b)``

   * - clip
     - ``clip(x, a, b)``
     - Clip: ``clip(x, min, max)``

Using Built-in Functions
-------------------------

Specifying Functions
^^^^^^^^^^^^^^^^^^^^^^

Select which built-in functions to use:

.. code-block:: python

   from featuristic import FeatureSynthesizer

   # Use all built-in functions (default)
   synth_all = FeatureSynthesizer(
       functions=None,  # None = all built-ins
       ...
   )

   # Use specific functions
   synth_specific = FeatureSynthesizer(
       functions=["add", "sub", "mul", "div", "sin", "cos"],
       ...
   )

   # Arithmetic only
   synth_arith = FeatureSynthesizer(
       functions=["add", "sub", "mul", "div", "square", "cube"],
       ...
   )

   # Trigonometric only
   synth_trig = FeatureSynthesizer(
       functions=["sin", "cos", "tan"],
       ...
   )

Example: Domain-Specific Function Set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For financial modeling (arithmetic + log):

.. code-block:: python

   financial_functions = [
       "add", "sub", "mul", "div",
       "log", "exp", "sqrt",
       "square", "cube"
   ]

   synth = FeatureSynthesizer(
       functions=financial_functions,
       n_features=10,
       generations=30,
       ...
   )

For signal processing (trigonometric):

.. code-block:: python

   signal_functions = [
       "sin", "cos", "tan",
       "add", "sub", "mul"
   ]

   synth = FeatureSynthesizer(
       functions=signal_functions,
       ...
   )

Adding Custom Functions in Rust
--------------------------------

**Note:** This requires:

1. Rust toolchain installed
2. Modifying source code in ``rust/featuristic-core/src/builtins.rs``
3. Rebuilding the extension with ``make build``

Step-by-Step Guide
^^^^^^^^^^^^^^^^^^^^

**1. Locate the built-ins file:**

.. code-block:: bash

   cd /path/to/featuristic
   ls rust/featuristic-core/src/builtins.rs

**2. Add your function to ``builtins.rs``:**

.. code-block:: rust

   // Example: Adding a "sigmoid" function
   pub fn sigmoid(x: f64) -> f64 {
       1.0 / (1.0 + (-x).exp())
   }

   // Register the function
   pub fn get_builtin(name: &str) -> Option<SymbolicOp> {
       match name {
           // ... existing functions ...
           "sigmoid" => Some(SymbolicOp {
               name: "sigmoid",
               op_id: 999,  // Unique ID
               arity: 1,     // Number of arguments
               eval: |args| {
                   let x = args[0];
                   Ok(vec![sigmoid(x)])
               },
           }),
           _ => None,
       }
   }

**3. Rebuild the extension:**

.. code-block:: bash

   make build

**4. Use your function:**

.. code-block:: python

   from featuristic import FeatureSynthesizer

   synth = FeatureSynthesizer(
       functions=["add", "mul", "sigmoid"],  # Your custom function!
       ...
   )

**Full Example: Adding a ``relu`` function**

.. code-block:: rust

   // In rust/featuristic-core/src/builtins.rs

   pub fn relu(x: f64) -> f64 {
       if x > 0.0 { x } else { 0.0 }
   }

   // Register in get_builtin()
   "relu" => Some(SymbolicOp {
       name: "relu",
       op_id: 1000,
       arity: 1,
       eval: |args| {
           let x = args[0];
           Ok(vec![relu(x)])
       },
   }),

Then rebuild and use:

.. code-block:: python

   synth = FeatureSynthesizer(
       functions=["relu", "add", "mul"],
       ...
   )

Best Practices for Custom Functions
------------------------------------

1. **Function Properties**
^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Arity**: Number of arguments (0, 1, or 2)
* **Domain**: Valid input range
* **Range**: Output range
* **Derivative**: Differentiability (for optimization)

2. **Error Handling**
^^^^^^^^^^^^^^^^^^^^

Protect against common errors:

.. code-block:: rust

   pub fn safe_log(x: f64) -> f64 {
       if x <= 0.0 {
           0.0  // Return safe default
       } else {
           x.ln()
       }
   }

   pub fn safe_sqrt(x: f64) -> f64 {
       if x < 0.0 {
           0.0  // Return safe default
       } else {
           x.sqrt()
       }
   }

3. **Performance Considerations**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Avoid expensive operations in tight loops
* Consider numerical stability
* Test with synthetic data first

4. **Documentation**
^^^^^^^^^^^^^^^^^^^^^^

Document your function thoroughly:

.. code-block:: rust

   /// Exponential moving average
   ///
   /// # Arguments
   /// * `x` - Current value
   /// * `alpha` - Smoothing factor (0, 1]
   ///
   /// # Returns
   /// EMA value
   ///
   /// # Example
   /// ```
   /// ema(1.0, 0.5) = 0.5
   /// ```
   pub fn ema(x: f64, alpha: f64) -> f64 {
       // Implementation
   }

Examples of Domain-Specific Functions
------------------------------------

Financial Functions
^^^^^^^^^^^^^^^^^^^

.. code-block:: rust

   // Black-Scholes-like function
   pub fn black_scholes_call(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
       let d1 = (s.ln() / k + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
       let d2 = d1 - sigma * t.sqrt();
       s * norm_cdf(d1) - k * (-r * t).exp() * norm_cdf(d2)
   }

Signal Processing
^^^^^^^^^^^^^^^^^^

.. code-block:: rust

   // Moving average
   pub fn moving_average(x: f64, window: f64) -> f64 {
       // Simplified: assumes pre-computed window
       x / window
   }

   // RMS (root mean square)
   pub fn rms(x: f64) -> f64 {
       (x * x).sqrt()
   }

Physics/Engineering
^^^^^^^^^^^^^^^^^^^^

.. code-block:: rust

   // Kinetic energy: KE = 0.5 * m * v²
   pub fn kinetic_energy(mass: f64, velocity: f64) -> f64 {
       0.5 * mass * velocity * velocity
   }

   // Potential energy: PE = m * g * h
   pub fn potential_energy(mass: f64, height: f64, gravity: f64) -> f64 {
       mass * gravity * height
   }

Testing Custom Functions
-----------------------

After adding a custom function:

**1. Unit test in Rust:**

.. code-block:: rust

   #[cfg(test)]
   mod tests {
       use super::*;

       #[test]
       fn test_sigmoid() {
           assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
           assert!((sigmoid(1.0) - 0.7310585786300049).abs() < 1e-10);
       }
   }

**2. Integration test in Python:**

.. code-block:: python

   import numpy as np
   from featuristic import FeatureSynthesizer

   # Create simple test data
   X = np.random.randn(100, 2)
   y = X[:, 0] * X[:, 1]  # Simple interaction

   # Test with custom function
   synth = FeatureSynthesizer(
       functions=["relu", "add", "mul"],  # Your custom function
       n_features=5,
       generations=10,
       random_state=42
   )

   X_aug = synth.fit_transform(X, y)

   # Check it works
   assert X_aug.shape[1] == 5

   # Inspect features
   programs = synth.get_programs()
   for prog in programs:
       print(prog['expression'])

**3. Verify correctness:**

.. code-block:: python

   import numpy as np
   from featuristic import random_tree, evaluate_tree

   # Generate a tree using your custom function
   tree = random_tree(
       depth=2,
       functions=["relu", "add"],
       feature_names=["x1", "x2"],
       seed=42
   )

   # Test evaluation
   X_test = pd.DataFrame({'x1': [1.0, -1.0, 0.0], 'x2': [0.5, 0.5, 0.5]})
   result = evaluate_tree(tree, X_test)

   print(f"Result: {result}")
   # Manually verify the calculation

Alternatives to Custom Functions
----------------------------------

If you can't modify Rust code, consider these alternatives:

**1. Post-processing in Python:**

.. code-block:: python

   # Synthesize features with built-ins
   synth = FeatureSynthesizer(
       functions=["add", "mul", "sqrt", "log"],
       ...
   )

   X_aug = synth.fit_transform(X_train, y_train)

   # Apply custom transformation in Python
   def custom_transform(X):
       return np.maximum(0, X)  # ReLU

   X_custom = custom_transform(X_aug)

**2. Feature unions:**

.. code-block:: python

   from sklearn.pipeline import FeatureUnion
   from sklearn.preprocessing import FunctionTransformer

   # Custom transformer
   def sigmoid_transform(X):
       return 1 / (1 + np.exp(-X))

   custom_transformer = FunctionTransformer(sigmoid_transform)

   # Combine with FeatureSynthesizer
   combined = FeatureUnion([
       ('synth', FeatureSynthesizer(...)),
       ('custom', custom_transformer)
   ])

**3. Custom objective function:**

.. code-block:: python

   # For FeatureSelector, use custom objective
   def custom_objective(X_subset, y):
       # Your custom logic
       model = YourCustomModel()
       model.fit(X_subset, y)
       return custom_scorer(y, model.predict(X_subset))

   selector = FeatureSelector(
       objective_function=custom_objective,
       ...
   )

Future Development
------------------

Planned features for easier custom function support:

* **Python-side registration** - Add functions without touching Rust
* **Plugin system** - Load functions from external modules
* **Function library** - Community-contributed function collection
* **JIT compilation** - Just-in-time compiled Python functions

For now, built-in functions cover most use cases. Request specific functions on GitHub issues!

What's Next
------------

* :doc:`feature_synthesis` - Using built-in functions
* :doc:`../api_reference/rust_functions` - Complete function list
* :doc:`../concepts/genetic_feature_synthesis` - How evolution works

Summary
-------

**Key points:**

1. **Built-in functions** cover most use cases (arithmetic, trigonometric, exponential, etc.)
2. **Custom functions** require Rust modification (advanced users only)
3. **Select functions** with the ``functions`` parameter
4. **Test thoroughly** after adding custom functions
5. **Alternatives exist** for Python-side transformations

**Quick reference:**

* **Arithmetic**: add, sub, mul, div, neg
* **Trigonometric**: sin, cos, tan
* **Exponential**: exp, log, sqrt
* **Power**: square, cube
* **Other**: abs, min, max, clip

For most users, the built-in functions are sufficient. Custom functions are only needed for highly specialized domains.
