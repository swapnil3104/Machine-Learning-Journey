# 📈 13. Linear Regression & Web Deployment

## 📌 Folder Overview
Welcome to the **Linear Regression** module! Linear Regression is the foundational parametric supervised learning algorithm used for predicting continuous numerical target variables. This folder covers Simple Linear Regression, Multiple Linear Regression, Polynomial Regression, mathematical assumptions, regression metrics, and building an end-to-end Flask web application for salary deployment.

### Why is this topic important?
Linear Regression provides unmatched model interpretability through feature coefficients. Understanding how Ordinary Least Squares (OLS) minimizes residual sums of squares is vital for mastering more complex machine learning models, regularization techniques, and regression diagnostics.

### What You Will Learn
- **Simple & Multiple Linear Regression**: Model equations ($y = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n$), Ordinary Least Squares (OLS), and matrix closed-form solutions $(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$.
- **Polynomial Regression**: Modeling non-linear relationships by creating polynomial feature transformations.
- **Assumptions Diagnostic**: Verifying Linearity, Independence, Homoscedasticity, Normality of Residuals, and Multicollinearity (VIF).
- **Regression Evaluation Metrics**: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), $R^2$ Score, and Adjusted $R^2$ Score.
- **End-to-End Projects & Web Deployment**: Training models on Medical Insurance and Salary datasets, saving pickled weights (`salary_model.pkl`), and deploying a interactive Flask web application.

---

## 📚 Topics Covered

Organized logically from basic line fitting to production deployment:

### 🟢 Beginner
- **Simple Linear Regression**: Fitting $y = m x + b$, minimizing Mean Squared Error (MSE), line visual intuition.
- **Regression Evaluation Metrics**:
  - MAE: $\frac{1}{n} \sum |y_i - \hat{y}_i|$
  - MSE: $\frac{1}{n} \sum (y_i - \hat{y}_i)^2$
  - RMSE: $\sqrt{\text{MSE}}$
  - $R^2$ Score & Adjusted $R^2$ Score (penalizing redundant features).

### 🟡 Intermediate
- **Multiple Linear Regression**: Closed-form mathematical derivation from scratch and Scikit-Learn implementation.
- **Polynomial Regression**: Modeling curved relationships using polynomial feature extensions ($x, x^2, x^3$).

### 🔴 Advanced
- **Assumptions of Linear Regression**:
  1. Linearity between features and target.
  2. Homoscedasticity (constant variance of residuals).
  3. Normality of Residuals (Q-Q plots & Shapiro-Wilk test).
  4. Absence of Multicollinearity (Variance Inflation Factor - VIF).
  5. Independence of errors (Durbin-Watson test).
- **Web App Deployment**:
  - Serializing models to `.pkl`.
  - Creating a Flask backend app (`app.py`), HTML forms (`templates/index.html`), and custom CSS styling (`static/style.css`).

| Module / Project | Subfolder / Files | Key Concepts | Level |
| :--- | :--- | :--- | :--- |
| **Simple / Multiple** | `Simple Multiple Regression/`, `Multiple Linear Regression/` | OLS derivation, multi-feature fitting. | Beginner → Intermediate |
| **Polynomial Regression** | `Polynomial Regression/` | Non-linear regression fitting. | Intermediate |
| **Metrics & Assumptions** | `Regression metrics/`, `Assumptions of Linear Regression.md` | $R^2$, MAE, RMSE, residual diagnostics. | Advanced |
| **End-to-End Projects** | `Project/` | Medical Insurance cost & Salary prediction. | Intermediate |
| **Flask Web App** | `Project/salary-prediction/` | Flask web deployment with `app.py` & HTML. | Production |

---

## 📂 File Structure

```text
13_Linear Regression/
├── README.md
├── Assumptions of Linear Regression.md
├── Assumptions_of_Linear_Regression.ipynb
├── Linear Regression Notes.md
├── Multiple Linear Regression/
│   ├── code-from-scratch.ipynb
│   └── multiple_linear_regression.ipynb
├── Polynomial Regression/
│   ├── Example1.ipynb
│   └── polynomial-regression.ipynb
├── Project/
│   ├── Medical Insurance Cost Prediction.ipynb
│   ├── Medical_Insurance_Model.pkl
│   ├── Salary Prediction Based on Experience.ipynb
│   ├── Salary_Data.csv
│   ├── insurance.csv
│   ├── salary_model.pkl
│   └── salary-prediction/
│       ├── app.py
│       ├── salary_model.pkl
│       ├── static/
│       │   └── style.css
│       └── templates/
│           └── index.html
├── Regression metrics/
│   ├── Example.ipynb
│   ├── Regression_Metrics_in_Machine_Learning.md
│   └── placement.csv
└── Simple Multiple Regression/
    ├── Example1.ipynb
    ├── Example2 .ipynb
    └── placement.csv
```
