# 🛡️ 16. Ridge Regression (L2 Regularization)

## 📌 Folder Overview
Welcome to the **Ridge Regression** module! Ridge Regression is an extension of Linear Regression that incorporates an L2 regularization penalty onto the loss function to prevent model overfitting, handle multicollinearity, and constrain coefficient magnitudes. This folder covers Ridge theory, mathematical derivations, closed-form analytical solvers from scratch, gradient descent optimization, and penalty hyperparameter tuning.

### Why is this topic important?
Ordinary Least Squares (OLS) regression easily overfits when features are highly correlated (multicollinearity) or when the number of features approaches or exceeds the sample size ($p \approx n$). Ridge Regression stabilizes parameter estimates by shrinking feature coefficients towards zero, drastically reducing variance while introducing slight bias.

### What You Will Learn
- **Ridge Cost Function**:
  $$J(\mathbf{w}) = \text{MSE} + \lambda \sum_{j=1}^{p} w_j^2 = \frac{1}{n} \sum_{i=1}^{n} (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \alpha \|\mathbf{w}\|_2^2$$
- **Closed-Form Solution**: Mathematical matrix derivation for Ridge parameters:
  $$\mathbf{w} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$
- **Gradient Descent for Ridge**: Updating weights using the modified gradient containing L2 penalty terms.
- **Hyperparameter $\alpha$ / $\lambda$ Tuning**: Visualizing coefficient shrinkage traces as $\alpha$ increases.

---

## 📚 Topics Covered

Organized logically from regularization intuition to custom mathematical implementations:

### 🟢 Beginner
- **Why Regularize?**: Limitations of OLS under high feature variance and collinearity.
- **Geometric Intuition**: L2 spherical constraint region intersecting OLS loss contours.

### 🟡 Intermediate
- **Ridge Shrinkage Effect**: Demonstrating how increasing $\alpha$ shrinks weights toward zero without setting them exactly to zero.
- **Scikit-Learn Implementation**: Training `Ridge()` models and tuning $\alpha$ via cross-validation (`RidgeCV`).

### 🔴 Advanced
- **Closed-Form Solver from Scratch**: Coding matrix solver $(\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$ from scratch in Python (`ridge-regression-from-scratch.ipynb`).
- **Gradient Descent Solver for Ridge**: Implementing iterative gradient updates with L2 regularization penalty (`ridge-regression-gradient-descent.ipynb`).

| Notebook | Focus Area | Implementation Method | Level |
| :--- | :--- | :--- | :--- |
| `Ridge Regularization.ipynb` | Scikit-Learn Ridge implementation and performance comparison. | Scikit-Learn API | Beginner → Intermediate |
| `ridge-regression-from-scratch.ipynb` | Closed-form matrix formula implementation from scratch. | NumPy Matrix Math | Advanced |
| `ridge-regression-from-scratch-m-and-b.ipynb` | 2D parameter ($m, b$) analytical formulation and derivation. | Analytical Math | Advanced |
| `ridge-regression-gradient-descent.ipynb` | Iterative optimization using Gradient Descent with L2 penalty. | Algorithmic from scratch | Advanced |
| `ridge-regression-key-understandings.ipynb` | Key takeaways, edge cases, and hyperparameter sensitivity. | Theoretical & Visual | Intermediate |

---

## 📂 File Structure

```text
16_Ridge Regression/
├── README.md
├── Ridge Regularization.ipynb
├── ridge-regression-from-scratch-m-and-b.ipynb
├── ridge-regression-from-scratch.ipynb
├── ridge-regression-gradient-descent.ipynb
└── ridge-regression-key-understandings.ipynb
```
