# ⚔️ 21. Support Vector Machines (SVM)

## 📌 Folder Overview
Welcome to the **Support Vector Machines (SVM)** module! SVM is a robust, versatile supervised learning algorithm capable of performing linear classification, non-linear classification, and regression. This folder covers maximum-margin hyperplanes, hard vs soft margins, support vectors, cost parameter $C$, and the Kernel Trick for non-linearly separable datasets.

### Why is this topic important?
SVMs excel in high-dimensional feature spaces (such as bioinformatics and text classification) where the number of dimensions exceeds the sample count. By maximizing the margin separating data classes, SVMs achieve exceptional generalization performance.

### What You Will Learn
- **Maximum-Margin Hyperplane**: Constructing decision hyperplanes $\mathbf{w}^T \mathbf{x} + b = 0$ that maximize the geometric margin $\frac{2}{\|\mathbf{w}\|}$ between support vectors.
- **Hard Margin vs Soft Margin**: Handling non-linearly separable datasets by introducing slack variables ($\xi_i$) and penalty parameter $C$.
- **Support Vectors**: Identifying the critical boundary data points that determine the location and orientation of the decision boundary.
- **The Kernel Trick**: Implicitly mapping input features into higher-dimensional feature spaces using kernel functions:
  - **Linear Kernel**: $K(\mathbf{x}, \mathbf{z}) = \mathbf{x}^T \mathbf{z}$
  - **Polynomial Kernel**: $K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T \mathbf{z} + c)^d$
  - **Radial Basis Function (RBF / Gaussian Kernel)**: $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2)$
- **Hyperparameter Tuning**: Tuning $C$ (cost of misclassification) and $\gamma$ (kernel influence radius).

---

## 📚 Topics Covered

Organized logically from margin maximization to non-linear kernel transformations:

### 🟢 Beginner
- **Hyperplanes & Decision Margins**: Understanding decision boundaries in 2D, 3D, and N-D space.
- **Linear SVM**: Hard-margin optimization formulation.

### 🟡 Intermediate
- **Soft-Margin SVM & Parameter $C$**:
  - Small $C$: Wider margin, tolerates misclassifications (higher bias, lower variance).
  - Large $C$: Narrow margin, strictly penalizes misclassifications (lower bias, higher variance).

### 🔴 Advanced
- **The Kernel Trick**:
  - Transforming 2D non-separable concentric rings into linearly separable 3D paraboloids.
  - Radial Basis Function (RBF) kernel and parameter $\gamma$ sensitivity.
- **Notebook Demos**: `Kernel Trick SVM.ipynb` and `SVM Demo.ipynb`.

| Notebook / Document | Focus Area | Key Concepts | Level |
| :--- | :--- | :--- | :--- |
| `SVM Demo.ipynb` | Linear and Soft-margin SVM demonstration. | Margins, support vectors, penalty $C$. | Intermediate |
| `Kernel Trick SVM.ipynb` | Non-linear classification via Kernel functions. | Polynomial & RBF kernels, $\gamma$ tuning. | Advanced |
| `Support_Vector_Machines_in_Machine_Learning.md` | Theoretical reference manual. | Convex optimization, Lagrange multipliers, math. | Reference |

---

## 📂 File Structure

```text
21_Support Vector Machines/
├── README.md
├── Kernel Trick SVM.ipynb
├── SVM Demo.ipynb
└── Support_Vector_Machines_in_Machine_Learning.md
```
