# 📘 Logistic Regression Hyperparameters — Complete Notes

## 📑 Table of Contents

1. [Introduction](#1-introduction)
2. [What Are Hyperparameters?](#2-what-are-hyperparameters)
3. [Important Logistic Regression Hyperparameters](#3-important-logistic-regression-hyperparameters)
4. [Penalty](#4-penalty)
5. [C — Regularization Strength](#5-c--regularization-strength)
6. [Solver](#6-solver)
7. [max_iter](#7-max_iter)
8. [tol](#8-tol)
9. [fit_intercept](#9-fit_intercept)
10. [class_weight](#10-class_weight)
11. [l1_ratio](#11-l1_ratio)
12. [multi_class](#12-multi_class)
13. [dual](#13-dual)
14. [intercept_scaling](#14-intercept_scaling)
15. [random_state](#15-random_state)
16. [warm_start](#16-warm_start)
17. [Hyperparameter Relationships](#17-hyperparameter-relationships)
18. [Hyperparameter Tuning](#18-hyperparameter-tuning)
19. [GridSearchCV Example](#19-gridsearchcv-example)
20. [RandomizedSearchCV Example](#20-randomizedsearchcv-example)
21. [Practical Tuning Strategy](#21-practical-tuning-strategy)
22. [Common Mistakes](#22-common-mistakes)
23. [Quick Reference Table](#23-quick-reference-table)
24. [Interview Questions](#24-interview-questions)
25. [Quick Revision](#25-quick-revision)

---

## 1. Introduction

**Logistic Regression** is a supervised learning algorithm commonly used for classification.

Although the basic mathematical model is relatively simple, its performance can be affected by choices such as:

- Regularization type
- Regularization strength
- Optimization solver
- Maximum number of iterations
- Convergence tolerance
- Class weighting
- Elastic-Net mixing

These are called **hyperparameters** because they are configured before model training rather than learned directly as model coefficients.

In scikit-learn, these options are exposed through `sklearn.linear_model.LogisticRegression`. The current documentation lists parameters such as `penalty`, `C`, `solver`, `max_iter`, `tol`, `class_weight`, `fit_intercept`, `l1_ratio`, and others. [Scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

---

## 2. What Are Hyperparameters?

A **hyperparameter** is a setting chosen before or around the training process that controls how an algorithm learns.

### Parameters vs Hyperparameters

| Parameters | Hyperparameters |
|---|---|
| Learned from training data | Set by the user or tuning process |
| Example: weights/coefficient | Example: `C` |
| Updated during optimization | Usually fixed for a training run |
| Model learns them | Practitioner chooses them |

For Logistic Regression:

```text
Training Data
     ↓
Hyperparameters
     ↓
Optimization
     ↓
Learned Coefficients
     ↓
Predictions
```

---

# 3. Important Logistic Regression Hyperparameters

The most important hyperparameters in scikit-learn Logistic Regression are:

| Hyperparameter | Main Purpose |
|---|---|
| `penalty` | Selects regularization type |
| `C` | Controls inverse regularization strength |
| `solver` | Selects optimization algorithm |
| `max_iter` | Maximum optimization iterations |
| `tol` | Convergence tolerance |
| `fit_intercept` | Whether to learn an intercept |
| `class_weight` | Assigns different class weights |
| `l1_ratio` | Controls L1/L2 mixture for Elastic-Net |
| `multi_class` | Controls multiclass formulation in versions where configurable |
| `dual` | Selects dual/primal formulation for supported configurations |
| `intercept_scaling` | Controls synthetic intercept feature for `liblinear` |
| `random_state` | Controls randomness where applicable |
| `warm_start` | Reuses previous solution |

Not every parameter applies to every solver.

---

# 4. Penalty

The `penalty` parameter determines the type of regularization applied to the model.

Common choices include:

```python
penalty="l1"
penalty="l2"
penalty="elasticnet"
penalty=None
```

### 4.1 L1 Regularization

L1 adds a penalty based on the absolute values of coefficients.

Conceptually:

```text
Loss + λ Σ|w|
```

Properties:

- Can drive some coefficients exactly to zero.
- Can perform a form of feature selection.
- Useful when a sparse model is desirable.

---

### 4.2 L2 Regularization

L2 adds a penalty based on squared coefficients.

Conceptually:

```text
Loss + λ Σw²
```

Properties:

- Shrinks coefficients toward zero.
- Usually does not make many coefficients exactly zero.
- Often a good general-purpose regularization choice.

---

### 4.3 Elastic-Net

Elastic-Net combines L1 and L2 regularization.

Conceptually:

```text
Loss + λ [α Σ|w| + (1-α) Σw²]
```

where `α` is represented by `l1_ratio` in scikit-learn.

```python
LogisticRegression(
    penalty="elasticnet",
    solver="saga",
    l1_ratio=0.5
)
```

---

### 4.4 No Regularization

```python
penalty=None
```

This removes the regularization penalty.

Use with care because regularization can help control model complexity and improve generalization.

---

# 5. C — Regularization Strength

`C` is one of the most important Logistic Regression hyperparameters.

### Key Rule

> **`C` is the inverse of regularization strength.**

Therefore:

```text
Small C → Stronger regularization
Large C → Weaker regularization
```

Scikit-learn documents `C` as the inverse of regularization strength. [Source](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### Example

```python
C=0.01
```

means stronger regularization than:

```python
C=1
```

and:

```python
C=100
```

### Typical Search Values

A logarithmic search is usually more useful than testing consecutive integers:

```python
C = [0.001, 0.01, 0.1, 1, 10, 100]
```

### Effect

| C | Regularization | Possible Result |
|---:|---|---|
| 0.001 | Very strong | Simpler model |
| 0.01 | Strong | More constrained coefficients |
| 0.1 | Moderate | More regularized |
| 1 | Moderate/default-style starting point | Baseline |
| 10 | Weak | More flexible |
| 100 | Very weak | Greater overfitting risk |

---

# 6. Solver

The `solver` specifies the numerical optimization algorithm used to fit Logistic Regression.

Common scikit-learn solvers include:

```text
lbfgs
liblinear
newton-cg
newton-cholesky
sag
saga
```

Scikit-learn currently documents `lbfgs` as the default solver and provides different solver capabilities depending on the penalty, dataset, and multiclass configuration. [Source](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

## Solver Comparison

| Solver | L1 | L2 | Elastic-Net | Large Dataset | Multiclass |
|---|---:|---:|---:|---:|---:|
| `lbfgs` | ❌ | ✅ | ❌ | Good | ✅ |
| `liblinear` | ✅ | ✅ | ❌ | Better for smaller datasets | Limited |
| `newton-cg` | ❌ | ✅ | ❌ | Good | ✅ |
| `newton-cholesky` | ❌ | ✅ | ❌ | Useful for certain large `n_samples`/feature settings | ✅ |
| `sag` | ❌ | ✅ | ❌ | Good for large datasets | ✅ |
| `saga` | ✅ | ✅ | ✅ | Good for large datasets | ✅ |

The exact compatibility depends on the scikit-learn version and configuration, so consult the current API documentation when selecting a solver. [Source](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### Practical Rule

Start with:

```python
solver="lbfgs"
```

for a conventional L2-regularized problem.

For L1:

```python
solver="liblinear"
```

or:

```python
solver="saga"
```

For Elastic-Net:

```python
solver="saga"
```

---

# 7. max_iter

`max_iter` specifies the maximum number of optimization iterations.

Default in current scikit-learn documentation:

```python
max_iter=100
```

[Source](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### Why Increase It?

If training produces a convergence warning, increase:

```python
max_iter=1000
```

or:

```python
max_iter=2000
```

Example:

```python
model = LogisticRegression(
    max_iter=1000
)
```

### Important

Increasing `max_iter` does not automatically improve generalization.

It mainly gives the optimizer more opportunity to converge.

---

# 8. tol

`tol` controls the stopping criterion for optimization.

Example:

```python
tol=1e-4
```

A smaller tolerance generally asks for a more precise convergence condition.

### Example

```python
tol=1e-3
```

may stop earlier than:

```python
tol=1e-6
```

### Trade-off

```text
Larger tol
   ↓
Earlier stopping
   ↓
Potentially faster

Smaller tol
   ↓
More precise convergence
   ↓
Potentially more computation
```

Do not confuse `tol` with model accuracy. It is an optimization convergence setting.

---

# 9. fit_intercept

Controls whether the model includes an intercept/bias term.

```python
fit_intercept=True
```

is the usual choice.

### If True

The model learns:

```text
z = w₁x₁ + w₂x₂ + ... + b
```

### If False

The model uses:

```text
z = w₁x₁ + w₂x₂ + ...
```

The intercept is important when the decision boundary does not need to pass through the origin.

Usually:

```python
fit_intercept=True
```

is appropriate unless there is a specific reason to remove the intercept.

---

# 10. class_weight

`class_weight` allows different classes to receive different weights during training.

Options include:

```python
class_weight=None
```

or:

```python
class_weight="balanced"
```

or a dictionary:

```python
class_weight={
    0: 1,
    1: 3
}
```

### Balanced Mode

Scikit-learn's `"balanced"` option automatically assigns weights inversely proportional to class frequencies:

```text
n_samples
---------------------------
n_classes × class_frequency
```

[Source](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### When Useful?

Suppose:

```text
Class 0 → 950 samples
Class 1 → 50 samples
```

A model could be dominated by the majority class.

Using:

```python
class_weight="balanced"
```

can give greater importance to the minority class.

### Important

Class weighting is not a substitute for evaluating the right metrics.

Use:

```text
Precision
Recall
F1-score
Confusion Matrix
ROC-AUC
PR-AUC
```

where appropriate.

---

# 11. l1_ratio

`l1_ratio` controls the mixture of L1 and L2 penalties when:

```python
penalty="elasticnet"
```

and the solver supports Elastic-Net, such as `saga`.

Range:

```text
0 ≤ l1_ratio ≤ 1
```

Interpretation:

| l1_ratio | Behavior |
|---:|---|
| 0 | L2-like |
| 0.25 | Mostly L2 |
| 0.5 | Equal mixture |
| 0.75 | Mostly L1 |
| 1 | L1-like |

Example:

```python
LogisticRegression(
    penalty="elasticnet",
    solver="saga",
    l1_ratio=0.5
)
```

Scikit-learn documents `l1_ratio` as the Elastic-Net mixing parameter. [Source](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

---

# 12. multi_class

The `multi_class` setting concerns how multiclass Logistic Regression is formulated.

Historically, common options included:

```text
ovr
multinomial
auto
```

### One-vs-Rest

For `K` classes, train `K` binary classifiers.

```text
Classifier 1 → Class 1 vs Rest
Classifier 2 → Class 2 vs Rest
Classifier 3 → Class 3 vs Rest
```

### Multinomial

A single model directly models the probabilities of all classes.

```text
Class 1
Class 2
Class 3
   ↓
Multinomial Logistic Regression
```

### Version Note

The handling of multiclass configuration has changed across scikit-learn releases. In current versions, supported solvers other than `liblinear` minimize the full multinomial loss for multiclass problems, while `liblinear` does not directly support the multinomial formulation. Check the version-specific documentation when writing production code. [Source](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

---

# 13. dual

`dual` chooses between the primal and dual optimization formulation for configurations that support it.

```python
dual=False
```

is the usual starting point.

The dual formulation is especially associated with `liblinear` and certain L2-regularized configurations.

### Rule of Thumb

For `liblinear`:

```text
n_samples > n_features
→ primal is generally preferred

n_features > n_samples
→ dual can be considered
```

Always verify compatibility with the selected solver and penalty.

---

# 14. intercept_scaling

`intercept_scaling` is mainly relevant to the `liblinear` solver when `fit_intercept=True`.

Scikit-learn internally represents the intercept using a synthetic feature in this configuration. That synthetic feature is regularized like other features, so increasing `intercept_scaling` can reduce the relative effect of regularization on the intercept. [Source](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

Example:

```python
LogisticRegression(
    solver="liblinear",
    intercept_scaling=10
)
```

For most users, this is not a first-line hyperparameter to tune.

---

# 15. random_state

`random_state` controls randomness where the selected solver uses it.

Example:

```python
random_state=42
```

In current scikit-learn documentation, it affects shuffling for:

```text
sag
saga
liblinear
```

and has no effect on other solvers. [Source](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

Using a fixed value helps make experiments reproducible when randomness is involved.

---

# 16. warm_start

When:

```python
warm_start=True
```

the solution from a previous `fit` call can be reused as initialization for the next fit for supported solvers.

Example:

```python
model = LogisticRegression(
    max_iter=1000,
    warm_start=True
)
```

This can be useful when fitting the same model repeatedly with related configurations or incremental experiments.

It is not a magic performance improvement for every workflow.

---

# 17. Hyperparameter Relationships

Some hyperparameters are directly related.

### `penalty` ↔ `solver`

Not every solver supports every penalty.

For example:

```text
Elastic-Net
    ↓
requires a compatible solver
    ↓
saga
```

### `C` ↔ Regularization

```text
C ↓ → Regularization ↑
C ↑ → Regularization ↓
```

### `max_iter` ↔ `tol`

These control optimization stopping behavior.

```text
max_iter → Maximum number of iterations
tol      → Convergence criterion
```

### `class_weight` ↔ Class Imbalance

```text
Imbalanced classes
       ↓
class_weight="balanced"
       ↓
Greater relative weight for minority classes
```

---

# 18. Hyperparameter Tuning

Hyperparameter tuning means systematically testing different configurations and selecting one using a validation strategy.

Common approaches:

### 1. Manual Search

```text
Try C = 0.01
Try C = 0.1
Try C = 1
Try C = 10
```

### 2. Grid Search

Tests all combinations in a specified grid.

```python
GridSearchCV
```

### 3. Randomized Search

Samples combinations from specified distributions.

```python
RandomizedSearchCV
```

### 4. Bayesian Optimization

Uses information from previous trials to choose subsequent configurations more strategically.

Grid and random search are standard approaches for exploring a hyperparameter search space. [Source](https://learn.microsoft.com/en-us/Azure/machine-Learning/how-to-tune-hyperparameters)

---

# 19. GridSearchCV Example

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

model = LogisticRegression(
    max_iter=1000
)

param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="f1"
)

grid_search.fit(X_train, y_train)

print("Best Parameters:")
print(grid_search.best_params_)

print("Best CV Score:")
print(grid_search.best_score_)
```

### Why `cv=5`?

The training data is divided into five folds.

```text
Fold 1 → Validation
Fold 2 → Training
Fold 3 → Training
Fold 4 → Training
Fold 5 → Training
```

Then the validation fold rotates.

This gives a more robust estimate than relying on a single split.

---

# 20. RandomizedSearchCV Example

```python
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear", "saga"],
    "class_weight": [None, "balanced"]
}

search = RandomizedSearchCV(
    estimator=LogisticRegression(max_iter=2000),
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    scoring="f1",
    random_state=42
)

search.fit(X_train, y_train)

print(search.best_params_)
```

Randomized search can be useful when the full combination space becomes large.

---

# 21. Practical Tuning Strategy

A practical workflow is:

### Step 1 — Start with a baseline

```python
model = LogisticRegression(
    max_iter=1000
)
```

### Step 2 — Scale Features

For many datasets, especially when feature magnitudes differ significantly:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scikit-learn documentation notes that `sag` and `saga` have convergence guarantees when features are approximately on the same scale. [Source](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### Step 3 — Tune `C`

Start with a logarithmic range:

```python
[0.001, 0.01, 0.1, 1, 10, 100]
```

### Step 4 — Tune penalty

Compare:

```text
L1
L2
Elastic-Net
```

where compatible.

### Step 5 — Choose an appropriate solver

Example:

```text
L2 → lbfgs
L1 → liblinear / saga
Elastic-Net → saga
```

### Step 6 — Address class imbalance

Consider:

```python
class_weight="balanced"
```

when justified by the dataset.

### Step 7 — Increase `max_iter` if needed

If convergence warnings occur:

```python
max_iter=2000
```

### Step 8 — Evaluate using suitable metrics

Do not optimize accuracy blindly.

Consider:

```text
Accuracy
Precision
Recall
F1
ROC-AUC
PR-AUC
Log Loss
```

depending on the problem.

---

# 22. Common Mistakes

## ❌ Mistake 1 — Thinking larger `C` means stronger regularization

This is incorrect.

```text
C ↓ → stronger regularization
C ↑ → weaker regularization
```

---

## ❌ Mistake 2 — Increasing `max_iter` to solve every problem

A convergence warning may have other causes, such as:

- Poor feature scaling
- Inappropriate solver
- Strongly correlated features
- Difficult optimization
- Extreme feature magnitudes

Increasing `max_iter` can help, but it is not always the underlying solution.

---

## ❌ Mistake 3 — Using incompatible penalty and solver

For example, Elastic-Net requires a compatible solver such as `saga` in scikit-learn.

---

## ❌ Mistake 4 — Scaling before splitting

Incorrect:

```python
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled)
```

This can leak information from the test set.

Better:

```python
X_train, X_test = train_test_split(X)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

## ❌ Mistake 5 — Choosing the best model using the test set

The test set should be kept for final evaluation.

Use:

```text
Training data → Model fitting
Validation/CV → Hyperparameter selection
Test data → Final evaluation
```

---

## ❌ Mistake 6 — Using accuracy for highly imbalanced data

Example:

```text
95% Class A
5% Class B
```

A model predicting Class A every time gets 95% accuracy but completely misses Class B.

Use class-specific metrics such as recall and F1 where appropriate.

---

# 23. Quick Reference Table

| Hyperparameter | Main Role | Common Starting Point |
|---|---|---|
| `penalty` | Regularization type | `"l2"` |
| `C` | Inverse regularization strength | `1.0` |
| `solver` | Optimization algorithm | `"lbfgs"` |
| `max_iter` | Maximum iterations | `100`–`1000` |
| `tol` | Convergence tolerance | `1e-4` |
| `fit_intercept` | Learn bias/intercept | `True` |
| `class_weight` | Class weighting | `None` |
| `l1_ratio` | Elastic-Net mixture | `0.5` when tuning |
| `multi_class` | Multiclass formulation | Version-dependent |
| `dual` | Primal/dual formulation | `False` |
| `intercept_scaling` | Intercept scaling for `liblinear` | `1.0` |
| `random_state` | Reproducibility where applicable | `42` |
| `warm_start` | Reuse previous solution | `False` |

---

# 24. Interview Questions

### Q1. What is the most important Logistic Regression hyperparameter?

`C` is commonly one of the most important because it controls the inverse strength of regularization.

---

### Q2. What happens when C decreases?

Regularization becomes stronger.

```text
C ↓ → Regularization ↑
```

---

### Q3. What is the default solver?

In current scikit-learn documentation, the default is:

```python
solver="lbfgs"
```

---

### Q4. Which solver supports Elastic-Net?

```python
solver="saga"
```

---

### Q5. What does `max_iter` control?

It specifies the maximum number of optimization iterations.

---

### Q6. What does `tol` control?

It controls the convergence stopping criterion.

---

### Q7. What does `class_weight="balanced"` do?

It automatically assigns class weights inversely related to class frequencies.

---

### Q8. What is the difference between L1 and L2 regularization?

```text
L1 → absolute coefficient penalty
L2 → squared coefficient penalty
```

L1 can produce sparse coefficients; L2 generally shrinks coefficients without forcing as many exactly to zero.

---

### Q9. Why use StandardScaler with Logistic Regression?

Scaling can improve optimization when features have very different magnitudes and is particularly important for certain solvers such as `sag` and `saga`.

---

### Q10. How do you tune Logistic Regression?

A typical process is:

```text
Define parameter grid
        ↓
Cross-validation
        ↓
GridSearchCV / RandomizedSearchCV
        ↓
Compare validation metric
        ↓
Select configuration
        ↓
Evaluate once on test set
```

---

# 25. Quick Revision

## 🔥 Logistic Regression Hyperparameter Cheat Sheet

```text
penalty
    ↓
Type of regularization

C
    ↓
Inverse of regularization strength

solver
    ↓
Optimization algorithm

max_iter
    ↓
Maximum optimization iterations

tol
    ↓
Convergence tolerance

fit_intercept
    ↓
Whether to learn intercept

class_weight
    ↓
Handle class imbalance

l1_ratio
    ↓
L1/L2 mixture for Elastic-Net

dual
    ↓
Primal vs dual formulation

random_state
    ↓
Reproducibility where randomness is used

warm_start
    ↓
Reuse previous solution
```

### ⭐ Most Important Relationships

```text
C ↓
→ Regularization ↑
→ Model becomes more constrained

C ↑
→ Regularization ↓
→ Model becomes less constrained
```

```text
L1
→ Sparse coefficients
→ Possible feature selection
```

```text
L2
→ Coefficient shrinkage
→ Common general-purpose choice
```

```text
Elastic-Net
→ L1 + L2
→ saga in scikit-learn
```

```text
Class Imbalance
→ class_weight="balanced"
→ Evaluate with appropriate class-sensitive metrics
```

## 🧠 One-Line Summary

> **Logistic Regression hyperparameters control regularization, optimization, convergence, class weighting, and model configuration; `C`, `penalty`, and `solver` are especially important when tuning the model.**

---

## 📚 References

- [Scikit-learn — LogisticRegression API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Scikit-learn — LogisticRegressionCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html)
- [Scikit-learn — Hyperparameter Search](https://scikit-learn.org/stable/modules/grid_search.html)
