# 📊 Assumptions of Linear Regression

## 📑 Table of Contents

1. [Introduction](#1-introduction)
2. [Why Are Assumptions Important?](#2-why-are-assumptions-important)
3. [Main Assumptions of Linear Regression](#3-main-assumptions-of-linear-regression)

   * [3.1 Linearity](#31-linearity)
   * [3.2 Independence of Observations](#32-independence-of-observations)
   * [3.3 Homoscedasticity](#33-homoscedasticity)
   * [3.4 Normality of Residuals](#34-normality-of-residuals)
   * [3.5 No Perfect Multicollinearity](#35-no-perfect-multicollinearity)
   * [3.6 No Significant Outliers](#36-no-significant-outliers)
4. [How to Check the Assumptions](#4-how-to-check-the-assumptions)
5. [Python Implementation](#5-python-implementation)
6. [What Happens When Assumptions Are Violated?](#6-what-happens-when-assumptions-are-violated)
7. [Assumption Checklist](#7-assumption-checklist)
8. [Summary](#8-summary)

---

# 1. Introduction

Linear Regression is a supervised machine learning algorithm used to predict a continuous target variable based on one or more independent variables.

The basic Simple Linear Regression equation is:

$$
y = \beta_0 + \beta_1x + \epsilon
$$

Where:

* $y$ = Dependent/target variable
* $x$ = Independent variable
* $\beta_0$ = Intercept
* $\beta_1$ = Regression coefficient/slope
* $\epsilon$ = Error term/residual

For Multiple Linear Regression:

$$
y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_nx_n + \epsilon
$$

Linear Regression works best when certain statistical assumptions are reasonably satisfied.

---

# 2. Why Are Assumptions Important?

The assumptions of Linear Regression help ensure that:

* The estimated coefficients are reliable.
* Predictions are meaningful.
* Statistical tests are valid.
* Confidence intervals are trustworthy.
* Model interpretation is appropriate.
* The regression model does not produce misleading conclusions.

> **Important:** Not every assumption is equally important for prediction.

For example, normality of residuals is mainly important for statistical inference, while linearity is important for obtaining a meaningful linear relationship.

---

# 3. Main Assumptions of Linear Regression

The major assumptions are:

| No. | Assumption                   | Main Purpose                                                           |
| --- | ---------------------------- | ---------------------------------------------------------------------- |
| 1   | Linearity                    | Relationship between X and Y should be approximately linear            |
| 2   | Independence                 | Observations/errors should be independent                              |
| 3   | Homoscedasticity             | Residual variance should remain approximately constant                 |
| 4   | Normality of Residuals       | Residuals should be approximately normally distributed                 |
| 5   | No Perfect Multicollinearity | Independent variables should not be perfectly correlated               |
| 6   | No Significant Outliers      | Extreme observations should not disproportionately influence the model |

---

# 3.1 Linearity 📈

## Definition

The relationship between the independent variables and the dependent variable should be approximately linear.

In Simple Linear Regression:

$$
y = \beta_0 + \beta_1x + \epsilon
$$

This means that changes in `X` should have a reasonably consistent relationship with changes in `Y`.

### Example

Suppose we want to predict salary based on years of experience.

```text
Experience → Salary

1 year  → ₹30,000
2 years → ₹35,000
3 years → ₹40,000
4 years → ₹45,000
```

This relationship is approximately linear.

### How to Check?

A scatter plot can be used:

```python
import matplotlib.pyplot as plt

plt.scatter(X, y)
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
```

You can also examine the residual plot.

### Violation

If the relationship looks like:

```text
       *
     *   *
   *       *
 *           *
```

instead of approximately following a straight-line pattern, Linear Regression may not be appropriate.

### Possible Solutions

* Polynomial Regression
* Feature transformation
* Log transformation
* Exponential transformation
* Decision Tree Regression
* Random Forest Regression

---

# 3.2 Independence of Observations 🔗

## Definition

Observations should be independent of one another.

In simple terms:

> One observation should not depend on another observation.

For example, suppose we have:

```text
Student 1 → 80 marks
Student 2 → 75 marks
Student 3 → 90 marks
Student 4 → 82 marks
```

If each student's result is independent, this assumption is reasonably satisfied.

---

## Example of Violation

Time-series data often violates independence.

For example:

```text
Monday    → ₹100
Tuesday   → ₹105
Wednesday → ₹110
Thursday  → ₹115
```

Today's value may depend on yesterday's value.

This creates **autocorrelation**.

### How to Check?

For time-series data, techniques such as:

* Durbin-Watson test
* Autocorrelation plots
* Residual analysis

can be used.

### Possible Solutions

* Time-series models
* ARIMA
* SARIMA
* Regression with lag features
* Generalized Least Squares

---

# 3.3 Homoscedasticity 📊

## Definition

Homoscedasticity means that the variance of the residuals should remain approximately constant across different levels of the predicted values.

In other words:

> The spread of errors should be roughly the same throughout the range of predictions.

### Good Example

```text
Residuals

  •   • •   •
 • • • • • • •
  •  • •  • •
-------------------- Predicted values
```

The residuals have approximately the same spread.

### Violation: Heteroscedasticity

A common violation looks like:

```text
Residuals

•
 ••
  •••
    ••••
       •••••
-------------------- Predicted values
```

The spread increases as predicted values increase.

This is called:

**Heteroscedasticity**

---

## How to Check?

Create a residual plot:

```python
import matplotlib.pyplot as plt

residuals = y_test - y_pred

plt.scatter(y_pred, residuals)
plt.axhline(y=0)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.show()
```

A random spread around zero is desirable.

---

## Statistical Test

One commonly used test is the **Breusch-Pagan test**.

```python
from statsmodels.stats.diagnostic import het_breuschpagan

test = het_breuschpagan(
    residuals,
    X_test
)

print(test)
```

---

## Possible Solutions

* Log transformation
* Square-root transformation
* Weighted Least Squares
* Robust standard errors
* Transforming the target variable

---

# 3.4 Normality of Residuals 🔔

## Definition

The residuals should be approximately normally distributed.

Residual:

$$
e_i = y_i - \hat{y}_i
$$

Where:

* $y_i$ = Actual value
* $\hat{y}_i$ = Predicted value
* $e_i$ = Residual

### Important

Normality applies to the **residuals**, not necessarily to the original independent variables.

---

## How to Check?

### Histogram

```python
import matplotlib.pyplot as plt

plt.hist(residuals, bins=20)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()
```

A roughly bell-shaped distribution is desirable.

---

## Q-Q Plot

A Q-Q plot is another common method.

```python
import statsmodels.api as sm
import matplotlib.pyplot as plt

sm.qqplot(residuals, line='45')
plt.show()
```

If the points approximately follow the diagonal line, residuals are reasonably close to normal.

---

## Statistical Tests

Common tests include:

* Shapiro-Wilk test
* Anderson-Darling test
* Jarque-Bera test

Example:

```python
from scipy.stats import shapiro

stat, p_value = shapiro(residuals)

print("Statistic:", stat)
print("P-value:", p_value)
```

### Interpretation

A common rule is:

```text
p-value > 0.05
→ No strong evidence against normality

p-value < 0.05
→ Evidence that residuals may not be normally distributed
```

> Do not rely only on a statistical test. For large datasets, even small deviations from normality can become statistically significant. Visual inspection is also important.

---

# 3.5 No Perfect Multicollinearity 🔄

## Definition

Multicollinearity occurs when independent variables are highly correlated with each other.

For example:

```text
X1 = House Size
X2 = Number of Rooms
X3 = Number of Bedrooms
```

These variables may contain very similar information.

---

## Why Is It a Problem?

High multicollinearity can make regression coefficients:

* Unstable
* Difficult to interpret
* Sensitive to small changes in data
* Associated with large standard errors

---

## Perfect Multicollinearity

Suppose:

$$
X_3 = X_1 + X_2
$$

Then `X3` can be completely determined from `X1` and `X2`.

This creates perfect multicollinearity.

---

## How to Check?

### Correlation Matrix

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.show()
```

High correlations between independent variables can indicate a potential problem.

---

## Variance Inflation Factor (VIF)

VIF is commonly used to detect multicollinearity.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = []

for i in range(X.shape[1]):
    vif.append(
        variance_inflation_factor(X.values, i)
    )

print(vif)
```

### General Interpretation

|  VIF | Interpretation          |
| ---: | ----------------------- |
|    1 | No multicollinearity    |
|  1–5 | Usually acceptable      |
| 5–10 | Potentially high        |
| > 10 | Often considered severe |

These are rules of thumb, not universal thresholds.

---

## Possible Solutions

* Remove highly correlated features.
* Combine related features.
* Use dimensionality reduction such as PCA.
* Use Ridge Regression.
* Use Lasso Regression.

---

# 3.6 No Significant Outliers 🎯

## Definition

Outliers are observations that are unusually far from the majority of the data.

Example:

```text
10
12
11
13
12
14
100  ← Outlier
```

The value `100` is very different from the other observations.

---

## Why Are Outliers a Problem?

Linear Regression uses the least-squares method.

The objective is:

$$
\min \sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

Because residuals are squared, large errors receive much more weight.

Therefore, extreme observations can strongly influence the regression line.

---

## How to Detect Outliers?

### Box Plot

```python
import matplotlib.pyplot as plt

plt.boxplot(df["Salary"])
plt.show()
```

### Z-Score

```python
from scipy.stats import zscore

z_scores = zscore(df["Salary"])
```

### IQR Method

$$
IQR = Q_3 - Q_1
$$

Lower boundary:

$$
Q_1 - 1.5(IQR)
$$

Upper boundary:

$$
Q_3 + 1.5(IQR)
$$

---

## Important

Not every outlier should be removed.

An outlier may represent:

* A genuine observation
* A rare but valid case
* A data-entry error
* Measurement error

Always investigate the reason before removing it.

---

# 4. How to Check the Assumptions 🔍

A practical workflow is:

```text
                 Linear Regression
                        ↓
                Train the Model
                        ↓
                 Generate Predictions
                        ↓
                 Calculate Residuals
                        ↓
        ┌───────────────┼────────────────┐
        ↓               ↓                ↓
     Linearity     Homoscedasticity   Normality
        ↓               ↓                ↓
   Scatter Plot    Residual Plot       Q-Q Plot
                        ↓
                 Check Multicollinearity
                        ↓
                      VIF
                        ↓
                  Check Outliers
                        ↓
                 Box Plot / Leverage
```

---

# 5. Python Implementation 🐍

A basic residual analysis can be performed using:

```python
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Predictions
y_pred = model.predict(X_test)

# Residuals
residuals = y_test - y_pred
```

---

## 5.1 Residual Plot

```python
plt.scatter(y_pred, residuals)

plt.axhline(y=0)

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")

plt.show()
```

### Desired Result

Residuals should be:

* Randomly distributed
* Centered around zero
* Without a clear pattern
* With approximately constant spread

---

## 5.2 Q-Q Plot

```python
sm.qqplot(residuals, line='45')

plt.title("Q-Q Plot of Residuals")
plt.show()
```

---

## 5.3 Histogram of Residuals

```python
plt.hist(residuals, bins=20)

plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals")

plt.show()
```

---

# 6. What Happens When Assumptions Are Violated? ⚠️

| Assumption Violated | Possible Problem                                          |
| ------------------- | --------------------------------------------------------- |
| Linearity           | Predictions may be systematically wrong                   |
| Independence        | Standard errors and inference can become unreliable       |
| Homoscedasticity    | Standard errors may be unreliable                         |
| Normality           | Hypothesis tests and confidence intervals may be affected |
| Multicollinearity   | Coefficients become unstable                              |
| Outliers            | Regression line can be strongly influenced                |

---

# 7. Assumption Checklist ✅

Before interpreting a Linear Regression model, check:

```text
☐ Is the relationship approximately linear?

☐ Are observations/errors independent?

☐ Are residuals randomly distributed?

☐ Is residual variance approximately constant?

☐ Are residuals approximately normally distributed?

☐ Is there serious multicollinearity?

☐ Are there influential outliers?

☐ Are the features correctly measured?

☐ Is the dataset appropriate for Linear Regression?
```

---

# 8. Important Distinction: Prediction vs Statistical Inference 🧠

One of the most important concepts is that the assumptions have different importance depending on the goal.

## For Prediction

If your primary goal is:

> "Predict house prices accurately."

Then the most important concerns include:

* Appropriate relationship between features and target
* Generalization to unseen data
* Outliers
* Data leakage
* Feature quality
* Model performance

Normality of residuals is usually less critical for obtaining predictions.

---

## For Statistical Inference

If your goal is:

> "Determine whether a feature has a statistically significant relationship with the target."

Then assumptions such as:

* Independence
* Homoscedasticity
* Appropriate model specification
* Residual behavior

become especially important.

---

# 9. Linear Regression Assumptions vs Machine Learning Practice 🤖

In traditional statistics, assumptions are often emphasized because the model is used for:

* Hypothesis testing
* Confidence intervals
* Coefficient interpretation
* Statistical inference

In machine learning, the focus is often more on:

* Test-set performance
* Cross-validation
* Generalization
* Feature engineering
* Prediction error

Therefore, a model can sometimes be useful for prediction even when some classical assumptions are not perfectly satisfied.

---

# 10. Quick Revision 📝

### 1️⃣ Linearity

```text
X and Y should have an approximately linear relationship.
```

### 2️⃣ Independence

```text
Observations/errors should be independent.
```

### 3️⃣ Homoscedasticity

```text
Residual variance should remain approximately constant.
```

### 4️⃣ Normality

```text
Residuals should be approximately normally distributed,
especially when performing statistical inference.
```

### 5️⃣ No Perfect Multicollinearity

```text
Independent variables should not be perfectly correlated.
```

### 6️⃣ No Significant Influential Outliers

```text
Extreme observations should not disproportionately affect
the regression model.
```

---

# 11. Final Summary 🎯

Linear Regression relies on several important assumptions:

$$
\boxed{
\text{Linearity}
+
\text{Independence}
+
\text{Homoscedasticity}
+
\text{Normality}
+
\text{No Perfect Multicollinearity}
+
\text{No Influential Outliers}
}
$$

The assumptions should not simply be memorized. You should learn **how to diagnose them, understand why violations matter, and choose an appropriate solution**.

### Remember

> **Linear Regression is not just about fitting a line — it is also about checking whether the data and residual behavior support the conclusions you want to make.**

---

## 📌 One-Line Memory Trick

**L-I-H-N-M-O**

```text
L → Linearity
I → Independence
H → Homoscedasticity
N → Normality
M → Multicollinearity (No Perfect Multicollinearity)
O → Outliers
```
