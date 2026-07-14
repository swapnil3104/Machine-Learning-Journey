# 🛠️ Feature Engineering in Machine Learning

> **Feature engineering** is the process of creating, transforming, selecting, and improving input features so that machine learning models can learn patterns more effectively.

---

## 🚀 Why Feature Engineering Matters

I remember building a model to improve **on-time delivery rates** for a **time-in-transit** project at my workplace.

Instead of training complex ensemble models, we used a **simple regression algorithm** with just **three additional engineered features** derived from existing data.

📈 **Result:**

| Before Feature Engineering | After Feature Engineering |
|----------------------------|---------------------------|
| On-Time Delivery = **48%** | On-Time Delivery = **56%** |

> 📊 Considering a dataset of **10 million records**, an 8% improvement is massive.

This demonstrates that **better features often outperform more complex models.**

---

# What is Feature Engineering?

Feature Engineering is the process of:

- Selecting useful features
- Creating new features
- Transforming existing features
- Removing irrelevant information
- Converting raw data into machine-readable inputs

Think of it like this:

```text
Raw Data
   │
   ▼
Feature Engineering
   │
   ▼
Better Features
   │
   ▼
Better Machine Learning Model
```

---

## Example

Consider a weather dataset.

| Temperature | Location | Date | Month | Year |
|-------------|----------|------|-------|------|
| 30°C | Mumbai | 15-06-2024 | June | 2024 |

If we're trying to capture **seasonality**, the **Month** already contains the important information.

The **Date** column doesn't add much value.

### Before

```
Temperature
Location
Date
Month
Year
```

### After

```
Temperature
Location
Month
Year
```

✅ Reduced dimensionality

✅ Faster training

✅ Less noise

---

# Types of Features

---

## 1️⃣ Numerical Features

Numerical features contain continuous numbers.

Examples:

- Age
- Height
- Weight
- Salary
- Temperature

Example

| Age | Salary |
|------|---------|
| 25 | 40000 |
| 32 | 55000 |
| 45 | 90000 |

---

## 2️⃣ Categorical Features

Contain discrete values.

Examples:

- Gender
- Country
- City
- Education
- Department

Example

| Name | Gender |
|------|---------|
| John | Male |
| Emma | Female |

### Types

### Binary

Only two categories

```
Yes / No

True / False

Male / Female
```

### Non-Binary

More than two categories

```
Red

Blue

Green

Yellow
```

---

## 3️⃣ Text Features

Contain natural language.

Examples

- Product Reviews
- Tweets
- Emails
- Comments

Example

```
"This phone has an amazing camera."
```

---

## 4️⃣ Time-Series Features

Data ordered over time.

Examples

- Stock prices
- Sales
- Weather
- Sensor readings

```
Jan → Feb → Mar → Apr → May
```

---

# 📊 Feature Engineering Workflow

```text
Raw Dataset
      │
      ▼
Handle Missing Values
      │
      ▼
Handle Outliers
      │
      ▼
Encode Categories
      │
      ▼
Scale Features
      │
      ▼
Create New Features
      │
      ▼
Select Best Features
      │
      ▼
Train Machine Learning Model
```

---

# Feature Engineering Techniques

---

# 1. Handling Missing Values

Missing values reduce model performance.

Example

| Age | Salary |
|------|---------|
| 25 | 40000 |
| NaN | 50000 |
| 30 | NaN |

---

## Method 1 — Imputation

Replace missing values.

Common techniques:

- Mean
- Median
- Mode

Example

```
Age

25

30

NaN

Mean = 27.5

Replace NaN → 27.5
```

---

## Method 2 — Deletion

Remove rows containing missing values.

Use when:

- Missing values < 10%
- Enough remaining data

---

# 2. Handling Outliers

Outliers are extremely large or small values.

Example

```
Salary

95K

102K

110K

98K

400K   ← Outlier
```

---

## Methods

### Replace

Replace with

- Max
- Min
- Median

---

### Transform

Apply

- Log Transformation
- Square Root

Example

```
400000

↓

log(400000)
```

---

### Robust Models

Use models less affected by outliers.

Examples

- Decision Trees
- Random Forest
- Gradient Boosting
- Ridge Regression

---

### Delete

Remove the outlier if appropriate.

---

# 3. Encoding Categorical Variables

Machine learning models understand numbers, not text.

---

## One-Hot Encoding

Original

| Gender |
|----------|
| Male |
| Female |
| Female |

↓

Encoded

| Female | Male |
|---------|------|
|0|1|
|1|0|
|1|0|

Visualization

```
Male

↓

[0 1]

Female

↓

[1 0]
```

---

## Label Encoding

Assign a unique number.

| Location | Label |
|-----------|-------|
| New York |1|
| California |2|
| Texas |3|

⚠️ Problem

Model may think

```
Texas > California > New York
```

which is incorrect.

---

## Ordinal Encoding

Used when categories have order.

| Education | Encoding |
|------------|----------|
| UG |1|
| PG |2|
| PhD |3|

Correct because

```
UG < PG < PhD
```

---

## Target Encoding

Replace category using target mean.

Example

| Location | Target |
|-----------|--------|
|NY|2|
|CA|3|
|TX|5|
|CA|1|
|TX|4|

Means

```
NY  → 2

CA  → 2

TX  → 4.5
```

---

# 4. Feature Scaling

Different scales confuse many ML algorithms.

Example

| Age | Income |
|------|---------|
|25|30000|
|45|150000|

Income dominates Age.

Scaling solves this.

---

## Normalization (Min-Max Scaling)

Range

```
0 → 1
```

Formula

```text
(x - min) / (max - min)
```

Visualization

```
Before

Age

20

30

40

60

↓

After

0

0.25

0.50

1.0
```

---

## Standardization (Z-score)

Mean becomes 0

Standard deviation becomes 1

Formula

```text
(x - mean) / std
```

Visualization

```
Before

10

20

30

40

↓

After

-1.2

-0.4

0.5

1.1
```

---

# 5. Creating New Features

Combine existing features to create more meaningful ones.

Example

House Dataset

| Length | Width |
|---------|-------|
|20|30|

↓

Create

```
Area = Length × Width

Area = 20 × 30 = 600
```

New dataset

| Length | Width | Area |
|---------|-------|------|
|20|30|600|

Area is much more useful for predicting price.

---

# 6. Feature Selection

Remove unnecessary columns.

Benefits

✅ Faster training

✅ Less overfitting

✅ Better accuracy

---

## Filter Methods

Use statistical techniques.

Examples

- Correlation
- Chi-Square
- ANOVA
- Information Gain

Visualization

```
Correlation Matrix

A ───── B  (0.95)

↓

Remove one feature
```

---

## Wrapper Methods

Try multiple feature combinations.

Examples

- Forward Selection
- Backward Elimination
- Recursive Feature Elimination (RFE)

Workflow

```
All Features

↓

Try Different Combinations

↓

Evaluate Model

↓

Choose Best Features
```

---

# Practical Example in Python

We'll use the famous **House Prices Dataset** from Kaggle.

Dataset Information

- 81 Columns
- Numerical Features
- Categorical Features
- Missing Values
- Outliers

Perfect for practicing feature engineering.

---

## Step 1

Download Dataset

```
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
```

---

## Step 2

Import Libraries

```python
import pandas as pd
import numpy as np
```

---

## Step 3

Load Dataset

```python
df = pd.read_csv("train.csv")
```

---

## Step 4

Explore Data

```python
df.head()
```

---

## Step 5

Check Missing Values

```python
df.isnull().sum()
```

---

## Step 6

Handle Missing Values

```python
df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())
```

---

## Step 7

Encode Categorical Variables

```python
df = pd.get_dummies(df)
```

---

## Step 8

Scale Features

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)
```

---

## Step 9

Train Model

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
```

---

# Best Practices

✅ Understand the dataset first

✅ Remove useless columns

✅ Handle missing values

✅ Treat outliers carefully

✅ Encode categorical variables properly

✅ Scale numerical features

✅ Create meaningful new features

✅ Select only important features

---

# Key Takeaways

- Feature Engineering is often more important than choosing a complex model.
- Good features help simple models outperform sophisticated algorithms.
- Proper handling of missing values, outliers, encoding, scaling, and feature selection significantly improves model performance.
- Creating domain-specific features can greatly enhance predictive power.
- Always experiment and validate engineered features using cross-validation.

---

# Summary Diagram

```text
                 RAW DATA
                     │
                     ▼
        Handle Missing Values
                     │
                     ▼
          Handle Outliers
                     │
                     ▼
      Encode Categorical Features
                     │
                     ▼
          Scale Numerical Features
                     │
                     ▼
         Create New Features
                     │
                     ▼
          Select Best Features
                     │
                     ▼
          Machine Learning Model
                     │
                     ▼
          Better Predictions 🚀
```

---

# References

- Scikit-learn Documentation
- Pandas Documentation
- Kaggle House Prices Dataset
- Feature Engineering for Machine Learning (Book)

---

**Author:** Your Notes  
**Topic:** Feature Engineering in Machine Learning  
**Difficulty:** Beginner → Intermediate