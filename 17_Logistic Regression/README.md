# 🎯 17. Logistic Regression & Classification

## 📌 Folder Overview
Welcome to the **Logistic Regression** module! Logistic Regression is the foundational supervised learning algorithm used for binary and multiclass classification problems. This folder covers the Perceptron trick, Sigmoid activation, Log Loss (Binary Cross-Entropy), Gradient Descent optimization, non-linear decision boundaries via Polynomial Features, Softmax Regression for multiclass targets, hyperparameter tuning, and an interactive Streamlit visualization application.

### Why is this topic important?
Classification is a core task in machine learning (e.g., spam detection, disease diagnosis, churn prediction). Unlike Linear Regression, which predicts continuous numbers, Logistic Regression outputs calibrated probabilities bounded between 0 and 1. Mastering Logistic Regression provides the groundwork for understanding Artificial Neural Networks and generalized linear models.

### What You Will Learn
- **Perceptron Trick**: Geometric intuition of updating decision lines based on misclassified points.
- **Sigmoid Activation Function**: Mapping arbitrary real values ($-\infty, +\infty$) to valid probabilities $(0, 1)$ via $\sigma(z) = \frac{1}{1 + e^{-z}}$.
- **Log Loss / Binary Cross-Entropy**: Mathematical derivation of the loss function optimized by classification models.
- **Polynomial Decision Boundaries**: Extending linear classifiers to fit non-linear circular or curved decision boundaries.
- **Softmax Regression (Multinomial Logistic Regression)**: Generalizing binary classification to multi-class target problems.
- **Streamlit Hyperparameter Tool**: Building an interactive web app (`streamlit-viz-tool.py`) to observe how solver choice, regularizers, and penalty hyperparameters alter decision boundaries in real time.

---

## 📚 Topics Covered

Organized logically from basic classification tricks to interactive visual tools:

### 🟢 Beginner
- **Classification vs Regression**: Why Linear Regression fails for categorical targets.
- **Perceptron Trick**: Updating line coefficients using learning rate $\eta$ when points are misclassified.
- **Sigmoid Function**: $\sigma(z) = \frac{1}{1 + e^{-z}}$, odds ratio, and log-odds transformation.

### 🟡 Intermediate
- **Log Loss Cost Function**:
  $$L(w) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
- **Gradient Descent for Classification**: Parameter update rules derived from Log Loss partial derivatives.
- **Polynomial Features in Logistic Regression**: Generating interaction features ($x_1^2, x_2^2, x_1 x_2$) to capture complex boundaries.

### 🔴 Advanced
- **Softmax Regression (Multiclass Classification)**: Softmax activation $\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum e^{z_k}}$ and Cross-Entropy Loss for multi-class labels.
- **Hyperparameter Optimization & Web Apps**:
  - Regularization strength parameter $C = \frac{1}{\lambda}$.
  - Solvers (`lbfgs`, `liblinear`, `saga`) and penalties ($L_1, L_2, \text{ElasticNet}$).
  - Interactive Streamlit dashboard (`streamlit-viz-tool.py`).

| Topic / Module | Subfolder / File | Key Concepts | Level |
| :--- | :--- | :--- | :--- |
| **Perceptron & Sigmoid** | `perceptron-trick.ipynb`, `perceptron-trick-sigmoid.ipynb` | Geometric line update & Sigmoid mapping. | Beginner |
| **Gradient Descent** | `gradient-descent.ipynb` | Log loss optimization via Gradient Descent. | Intermediate |
| **Polynomial Features** | `Polynomial Features in Logistic Regression/` | Non-linear boundary generation. | Intermediate |
| **Softmax Regression** | `Softmax Regression/Softmax_Regression_Notes.md` | Multi-class cross-entropy classification. | Advanced |
| **Streamlit Viz Tool** | `Logistic Regression Hyperparameters/` | Interactive web dashboard for hyperparameter tuning. | Advanced / App |

---

## 📂 File Structure

```text
17_Logistic Regression/
├── README.md
├── Notes.md
├── gradient-descent.ipynb
├── perceptron-trick-sigmoid.ipynb
├── perceptron-trick.ipynb
├── Logistic Regression Hyperparameters/
│   ├── Logistic_Regression_Hyperparameters.md
│   └── streamlit-viz-tool.py
├── Polynomial Features in Logistic Regression/
│   └── polynomial-logistic-regression.ipynb
└── Softmax Regression/
    └── Softmax_Regression_Notes.md
```
