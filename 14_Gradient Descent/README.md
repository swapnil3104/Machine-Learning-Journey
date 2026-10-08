# 📉 14. Gradient Descent Optimization Engine

## 📌 Folder Overview
Welcome to the **Gradient Descent** module! Gradient Descent is the foundational iterative optimization algorithm used to train machine learning models and neural networks by minimizing a cost function. This folder covers mathematical derivations, step-by-step algorithms built from scratch, 3D loss surface visualizations, animated convergence plots, and the three main variants of Gradient Descent.

### Why is this topic important?
While analytical closed-form solutions (like OLS in Linear Regression) work for small datasets, they become computationally prohibitive ($O(n^3)$ matrix inversion) as feature dimensions grow. Gradient Descent provides an efficient, scalable optimization engine that powers Logistic Regression, Support Vector Machines, Neural Networks, and Deep Learning architectures.

### What You Will Learn
- **Mathematical Foundations**: Derivative slopes, partial derivatives ($\frac{\partial L}{\partial w}, \frac{\partial L}{\partial b}$), and parameter update rules ($w_{new} = w_{old} - \alpha \frac{\partial L}{\partial w}$).
- **Learning Rate ($\alpha$) Dynamics**: Impact of small vs excessively large learning rates (slow convergence vs exploding divergence).
- **Types of Gradient Descent**:
  - **Batch Gradient Descent**: Updating weights using the entire training dataset per epoch.
  - **Stochastic Gradient Descent (SGD)**: Updating weights per single sample, adding stochastic noise to escape local minima.
  - **Mini-Batch Gradient Descent**: Updating weights over small mini-batches (e.g., 32, 64 samples), combining speed and stability.
- **Visualizations**: 2D animated trajectory plots, 3D loss surfaces, contour plots, and GIF animations (`stochastic_animation_contour_plot.gif`).

---

## 📚 Topics Covered

Organized logically from mathematical formulas to multi-variant animations:

### 🟢 Beginner
- **Intuition of Hill-Descent**: Cost functions $J(w, b)$, slope direction, and step sizes.
- **Learning Rate Tuning**: Visualizing convergence behavior across different $\alpha$ values.

### 🟡 Intermediate
- **Mathematical Derivation**: Calculus derivations for weights ($w$) and intercept ($b$).
- **Coding GD from Scratch**: Building custom Python classes for Gradient Descent step-by-step without Scikit-Learn.

### 🔴 Advanced
- **Gradient Descent Variants**:
  - **Batch GD**: Smooth monotonic convergence, high memory footprint.
  - **Stochastic GD**: Fast iterations, noisy trajectory, ideal for massive datasets.
  - **Mini-Batch GD**: Optimal GPU efficiency, balanced convergence trajectory.
- **3D Surface & Contour Animations**: Generating loss landscape trajectories and GIF animations (`mini_batch_contour_plot.gif`).

| Concept / Notebook | Description | Output Format | Level |
| :--- | :--- | :--- | :--- |
| `gradient-descent-code-from-scratch.ipynb` | Writing custom GD algorithm from scratch in Python. | Code Implementation | Intermediate |
| `gradient-descent-3d.ipynb` | Plotting 3D cost function surfaces and gradient vectors. | 3D Visualization | Advanced |
| `gradient-descent-animation(both-m-and-b).ipynb` | Animated line fitting progression per iteration step. | Matplotlib Animation | Advanced |
| `Types of Gradient Descent/` | Batch, SGD, and Mini-Batch comparison notebooks & GIFs. | Notebooks + GIFs | Advanced |

---

## 📂 File Structure

```text
14_Gradient Descent/
├── README.md
├── Gradient_Descent_Formula_Derivation.md
├── Notes.md
├── gradient-descent-3d.ipynb
├── gradient-descent-animation(both-m-and-b).ipynb
├── gradient-descent-animation(onlyb).ipynb
├── gradient-descent-code-from-scratch.ipynb
├── gradient_descent_step_by_step.ipynb
└── Types of Gradient Descent/
    ├── batch-gradient-descent.ipynb
    ├── mini-batch-gradient-descent-from-scratch.ipynb
    ├── mini_batch_contour_plot.gif
    ├── stochastic-gradient-descent-animation.ipynb
    ├── stochastic-gradient-descent-from-scratch.ipynb
    ├── stochastic_animation_contour_plot.gif
    ├── stochastic_animation_cost_plot.gif
    └── stochastic_animation_line_plot.gif
```
