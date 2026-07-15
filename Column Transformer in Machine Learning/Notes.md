# 🔀 Column Transformer in Machine Learning

> **ColumnTransformer** is a preprocessing utility in **Scikit-learn** that allows you to apply **different preprocessing techniques to different columns** of a dataset within a single pipeline.

It is one of the most useful tools when working with **mixed data types** (numerical + categorical + text).

---

# 🎯 Why Do We Need ColumnTransformer?

Real-world datasets usually contain different types of features.

Example:

| Age | Salary | Gender | City |
|-----|---------|--------|------|
| 25 | 45000 | Male | Mumbai |
| 32 | 62000 | Female | Pune |
| 45 | 85000 | Male | Delhi |

Different columns require different preprocessing.

| Column Type | Required Preprocessing |
|--------------|------------------------|
| Age | Standard Scaling |
| Salary | Standard Scaling |
| Gender | One-Hot Encoding |
| City | One-Hot Encoding |

Without a ColumnTransformer, we would need to preprocess each column separately.

---

# 📌 What is ColumnTransformer?

A **ColumnTransformer** applies different transformations to different subsets of columns **at the same time**.

```text
                 Dataset
                    │
                    ▼
          ColumnTransformer
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
 Numerical Columns      Categorical Columns
 StandardScaler()      OneHotEncoder()
          │                   │
          └─────────┬─────────┘
                    ▼
          Transformed Dataset
```

---

# 🚀 Advantages

✅ Handles mixed data types

✅ Cleaner code

✅ Prevents preprocessing mistakes

✅ Works perfectly with Pipelines

✅ Easy to deploy in production

---

# Basic Syntax

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["Age", "Salary"]),
        ("cat", OneHotEncoder(), ["Gender", "City"])
    ]
)
```

---

# Understanding the Syntax

```python
ColumnTransformer(
    transformers=[
        ("name", transformer, columns)
    ]
)
```

Each transformer consists of:

| Parameter | Description |
|-----------|-------------|
| name | Name of the transformer |
| transformer | Preprocessing method |
| columns | Columns to apply the transformation |

---

# Example Dataset

| Age | Salary | Gender | City |
|-----|---------|--------|------|
|25|45000|Male|Mumbai|
|30|60000|Female|Delhi|
|40|85000|Male|Pune|

---

# Step 1 — Import Libraries

```python
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
```

---

# Step 2 — Create Dataset

```python
data = {
    "Age":[25,30,40],
    "Salary":[45000,60000,85000],
    "Gender":["Male","Female","Male"],
    "City":["Mumbai","Delhi","Pune"]
}

df = pd.DataFrame(data)
```

---

# Step 3 — Define ColumnTransformer

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["Age","Salary"]),
        ("cat", OneHotEncoder(), ["Gender","City"])
    ]
)
```

---

# Step 4 — Transform Dataset

```python
X = preprocessor.fit_transform(df)
```

---

# What Happens Internally?

```text
Original Dataset

Age
Salary
Gender
City

        │
        ▼

ColumnTransformer

        │

 ┌───────────────┬────────────────┐
 │               │                │
 ▼               ▼                ▼
Age          Salary        Gender, City
 │               │                │
 ▼               ▼                ▼
StandardScaler StandardScaler OneHotEncoder
 │               │                │
 └───────────────┴────────────────┘
                 │
                 ▼
        Final Transformed Data
```

---

# Example Output

Original Dataset

| Age | Salary | Gender | City |
|-----|---------|--------|------|
|25|45000|Male|Mumbai|

↓

After Transformation

| Age | Salary | Gender_Female | Gender_Male | City_Delhi | City_Mumbai | City_Pune |
|-----|---------|---------------|-------------|------------|-------------|-----------|
|-1.22|-1.12|0|1|0|1|0|

Notice:

- Numerical columns are scaled.
- Categorical columns are one-hot encoded.

---

# Using Different Transformers Together

You can combine multiple preprocessing techniques.

```python
from sklearn.impute import SimpleImputer

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]),
            ["Age","Salary"]
        ),

        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder())
            ]),
            ["Gender","City"]
        )
    ]
)
```

---

# Visual Workflow

```text
                   Raw Dataset
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
 Numerical Columns              Categorical Columns
        │                               │
        ▼                               ▼
 Missing Value Imputation      Missing Value Imputation
        │                               │
        ▼                               ▼
 StandardScaler()             OneHotEncoder()
        │                               │
        └───────────────┬───────────────┘
                        ▼
               Combined Features
                        │
                        ▼
              Machine Learning Model
```

---

# ColumnTransformer with Pipeline

One of the biggest advantages is integrating preprocessing and model training.

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

pipeline.fit(X_train, y_train)
```

Workflow:

```text
Training Data
      │
      ▼
ColumnTransformer
      │
      ▼
Preprocessed Data
      │
      ▼
Linear Regression
      │
      ▼
Predictions
```

---

# Common Parameters

```python
ColumnTransformer(
    transformers=[],
    remainder="drop"
)
```

| Parameter | Description |
|-----------|-------------|
| transformers | List of preprocessing steps |
| remainder="drop" | Drop remaining columns (default) |
| remainder="passthrough" | Keep untouched columns |

---

## Example: `remainder="passthrough"`

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(), ["Gender"])
    ],
    remainder="passthrough"
)
```

Here, only **Gender** is encoded, while all other columns remain unchanged.

---

# When Should You Use ColumnTransformer?

Use it when your dataset contains:

- ✅ Numerical columns
- ✅ Categorical columns
- ✅ Date features
- ✅ Text features
- ✅ Different preprocessing requirements for different columns

---

# Advantages

| Benefit | Explanation |
|----------|-------------|
| ✔ Clean Code | No manual preprocessing |
| ✔ Reusable | Same preprocessing during training and testing |
| ✔ Faster Development | Single preprocessing object |
| ✔ Production Ready | Prevents data leakage |
| ✔ Easy Integration | Works seamlessly with Pipelines |

---

# Best Practices

✅ Separate numerical and categorical columns.

✅ Combine with `Pipeline`.

✅ Handle missing values before scaling or encoding.

✅ Use `remainder="passthrough"` if you want to keep untouched columns.

✅ Always `fit()` on training data and `transform()` on test data.

---

# Summary Diagram

```text
                      DATASET
                         │
      ┌──────────────────┴──────────────────┐
      │                                     │
      ▼                                     ▼
 Numerical Features                Categorical Features
      │                                     │
      ▼                                     ▼
 SimpleImputer()                  SimpleImputer()
      │                                     │
      ▼                                     ▼
 StandardScaler()                OneHotEncoder()
      │                                     │
      └──────────────────┬──────────────────┘
                         ▼
                ColumnTransformer
                         │
                         ▼
                 Feature Matrix (X)
                         │
                         ▼
               Machine Learning Model
```

---

# 📚 Key Takeaways

- **ColumnTransformer** applies different preprocessing techniques to different columns in one step.
- It is ideal for datasets containing a mix of numerical and categorical features.
- It integrates seamlessly with **Pipeline**, making workflows cleaner and reducing the risk of data leakage.
- Common transformations include **StandardScaler** for numerical data and **OneHotEncoder** for categorical data.
- It is considered a best practice for building scalable and production-ready machine learning pipelines.