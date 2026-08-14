# 🚀 Machine Learning Pipelines A-Z

> A **Machine Learning Pipeline** is a sequence of data processing and modeling steps that automates the entire machine learning workflow, from raw data to predictions.

Using pipelines ensures that **the same preprocessing steps are applied consistently during both training and prediction**, reducing errors and preventing data leakage.

---

# 📌 Why Do We Need Pipelines?

In a real-world machine learning project, data goes through several preprocessing steps before it reaches the model.

Without a pipeline, we have to perform each step manually.

### Without Pipeline

```text
Raw Data
   │
   ▼
Handle Missing Values
   │
   ▼
Encode Categories
   │
   ▼
Scale Features
   │
   ▼
Feature Selection
   │
   ▼
Train Model
```

Problems:

❌ Repetitive code

❌ Easy to make mistakes

❌ Data leakage

❌ Difficult to maintain

---

### With Pipeline

```text
Raw Data
    │
    ▼
Pipeline
    │
    ▼
Preprocessing
    │
    ▼
Machine Learning Model
    │
    ▼
Prediction
```

Everything is automated.

---

# 🎯 What is a Pipeline?

A **Pipeline** chains multiple preprocessing steps and the machine learning model into a single object.

Instead of writing multiple lines of preprocessing code, we create one workflow.

```text
Dataset
   │
   ▼
Pipeline
   │
   ├── Missing Value Handling
   ├── Feature Scaling
   ├── Encoding
   ├── Feature Selection
   └── Model Training
   │
   ▼
Prediction
```

---

# 🧩 Components of a Machine Learning Pipeline

```text
                Machine Learning Pipeline

                        │
      ┌─────────────────┼─────────────────┐
      │                 │                 │
      ▼                 ▼                 ▼
 Data Cleaning    Feature Engineering    Model
      │                 │                 │
      └─────────────────┼─────────────────┘
                        ▼
                  Model Evaluation
                        │
                        ▼
                   Final Prediction
```

---

# Typical Pipeline Workflow

```text
Raw Dataset
     │
     ▼
Train-Test Split
     │
     ▼
Handle Missing Values
     │
     ▼
Encode Categorical Features
     │
     ▼
Scale Numerical Features
     │
     ▼
Feature Selection (Optional)
     │
     ▼
Train Machine Learning Model
     │
     ▼
Evaluate Performance
     │
     ▼
Make Predictions
```

---

# Step 1: Import Libraries

```python
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
```

---

# Step 2: Create Sample Dataset

```python
data = {
    "Age":[25,30,40],
    "Salary":[45000,60000,85000],
    "Gender":["Male","Female","Male"],
    "City":["Mumbai","Delhi","Pune"],
    "Target":[1,0,1]
}

df = pd.DataFrame(data)
```

---

# Step 3: Separate Features

```python
X = df.drop("Target", axis=1)

y = df["Target"]
```

---

# Step 4: Define Numerical Pipeline

```python
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
```

Workflow

```text
Numerical Columns

Age
Salary

      │
      ▼
Missing Value Imputer

      │
      ▼
StandardScaler

      │
      ▼
Processed Features
```

---

# Step 5: Define Categorical Pipeline

```python
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder())
])
```

Workflow

```text
Categorical Columns

Gender
City

      │
      ▼
Missing Value Imputer

      │
      ▼
OneHotEncoder

      │
      ▼
Processed Features
```

---

# Step 6: Combine Using ColumnTransformer

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, ["Age","Salary"]),
        ("cat", categorical_pipeline, ["Gender","City"])
    ]
)
```

Visualization

```text
                    Dataset
                       │
       ┌───────────────┴───────────────┐
       │                               │
       ▼                               ▼
Numerical Columns              Categorical Columns
       │                               │
       ▼                               ▼
Numerical Pipeline          Categorical Pipeline
       │                               │
       └───────────────┬───────────────┘
                       ▼
               Combined Features
```

---

# Step 7: Create Final Pipeline

```python
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])
```

Workflow

```text
Dataset

   │

   ▼

ColumnTransformer

   │

   ▼

Linear Regression

   │

   ▼

Predictions
```

---

# Step 8: Train Pipeline

```python
pipeline.fit(X, y)
```

Everything happens automatically.

```text
fit()

↓

Missing Value Handling

↓

Scaling

↓

Encoding

↓

Model Training
```

---

# Step 9: Predict

```python
predictions = pipeline.predict(X)
```

Prediction Workflow

```text
New Data

↓

Pipeline

↓

Imputation

↓

Scaling

↓

Encoding

↓

Prediction
```

---

# Complete Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder())
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, ["Age","Salary"]),
    ("cat", cat_pipeline, ["Gender","City"])
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

pipeline.fit(X, y)
```

---

# Pipeline Architecture

```text
                      Raw Dataset
                           │
                           ▼
                    Train-Test Split
                           │
                           ▼
                  ColumnTransformer
          ┌────────────────┴────────────────┐
          │                                 │
          ▼                                 ▼
 Numerical Pipeline              Categorical Pipeline
          │                                 │
          ▼                                 ▼
 SimpleImputer()                 SimpleImputer()
          │                                 │
          ▼                                 ▼
 StandardScaler()               OneHotEncoder()
          │                                 │
          └────────────────┬────────────────┘
                           ▼
                    Feature Matrix
                           │
                           ▼
                Machine Learning Model
                           │
                           ▼
                     Predictions
```

---

# Common Pipeline Methods

| Method | Purpose |
|---------|----------|
| `fit()` | Learn preprocessing parameters and train the model |
| `transform()` | Apply preprocessing only |
| `fit_transform()` | Fit and transform together |
| `predict()` | Make predictions |
| `predict_proba()` | Predict probabilities (classification) |
| `score()` | Evaluate the model |

---

# Advantages of Pipelines

✅ Cleaner code

✅ Easy to read

✅ Prevents data leakage

✅ Reproducible workflow

✅ Easier deployment

✅ Combines preprocessing and model

✅ Works with GridSearchCV

✅ Easy hyperparameter tuning

---

# Pipeline vs Manual Workflow

| Manual Workflow | Pipeline Workflow |
|-----------------|------------------|
| Multiple preprocessing steps | One unified workflow |
| More code | Less code |
| Risk of forgetting steps | Automatic execution |
| Higher chance of data leakage | Prevents data leakage |
| Harder to maintain | Easier to maintain |

---

# Best Practices

✔ Always split data before fitting the pipeline.

✔ Use **ColumnTransformer** for mixed data types.

✔ Keep preprocessing inside the pipeline.

✔ Train only on the training data.

✔ Save the entire pipeline using `joblib` or `pickle`.

✔ Use the same pipeline for training and inference.

---

# Real-World Pipeline Example

```text
Customer Churn Dataset

           │
           ▼
Train-Test Split

           │
           ▼
Missing Value Imputation

           │
           ▼
One-Hot Encoding

           │
           ▼
Feature Scaling

           │
           ▼
Feature Selection

           │
           ▼
Random Forest Classifier

           │
           ▼
Customer Churn Prediction
```

---

# Summary Diagram

```text
                 MACHINE LEARNING PIPELINE

                     Raw Dataset
                          │
                          ▼
                  Train-Test Split
                          │
                          ▼
               Data Preprocessing
        ┌──────────────┬──────────────┐
        │              │              │
        ▼              ▼              ▼
 Missing Values   Encoding      Feature Scaling
        │              │              │
        └──────────────┼──────────────┘
                       ▼
              Feature Engineering
                       │
                       ▼
          Feature Selection (Optional)
                       │
                       ▼
          Machine Learning Algorithm
                       │
                       ▼
                 Model Evaluation
                       │
                       ▼
                  Final Prediction
```

---

# 📚 Key Takeaways

- A **Machine Learning Pipeline** automates the complete ML workflow.
- It combines **preprocessing**, **feature engineering**, and **model training** into one reusable object.
- Pipelines help prevent **data leakage** by ensuring preprocessing is learned only from the training data.
- They integrate seamlessly with **ColumnTransformer**, **GridSearchCV**, and cross-validation.
- Pipelines make ML code cleaner, reproducible, scalable, and production-ready.