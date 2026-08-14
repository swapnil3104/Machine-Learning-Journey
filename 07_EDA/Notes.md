# Exploratory Data Analysis (EDA) — Complete Notes

## Table of Contents

1. [Introduction to EDA](#1-introduction-to-eda)
2. [Why EDA is Important](#2-why-eda-is-important)
3. [EDA Workflow](#3-eda-workflow)
4. [Types of Data](#4-types-of-data)
5. [Understanding the Dataset](#5-understanding-the-dataset)
6. [Loading Data with Pandas](#6-loading-data-with-pandas)
7. [Initial Data Inspection](#7-initial-data-inspection)
8. [Data Types](#8-data-types)
9. [Missing Values](#9-missing-values)
10. [Duplicate Values](#10-duplicate-values)
11. [Statistical Summary](#11-statistical-summary)
12. [Univariate Analysis](#12-univariate-analysis)
13. [Bivariate Analysis](#13-bivariate-analysis)
14. [Multivariate Analysis](#14-multivariate-analysis)
15. [Categorical Data Analysis](#15-categorical-data-analysis)
16. [Numerical Data Analysis](#16-numerical-data-analysis)
17. [Outlier Detection](#17-outlier-detection)
18. [Correlation Analysis](#18-correlation-analysis)
19. [Covariance](#19-covariance)
20. [Distribution Analysis](#20-distribution-analysis)
21. [Skewness](#21-skewness)
22. [Kurtosis](#22-kurtosis)
23. [Feature Relationships](#23-feature-relationships)
24. [GroupBy Analysis](#24-groupby-analysis)
25. [Pivot Tables](#25-pivot-tables)
26. [Visualization for EDA](#26-visualization-for-eda)
27. [Common EDA Plots](#27-common-eda-plots)
28. [Data Quality Checks](#28-data-quality-checks)
29. [Feature Engineering During EDA](#29-feature-engineering-during-eda)
30. [Target Variable Analysis](#30-target-variable-analysis)
31. [EDA for Machine Learning](#31-eda-for-machine-learning)
32. [EDA Checklist](#32-eda-checklist)
33. [Common Mistakes](#33-common-mistakes)
34. [Practical EDA Template](#34-practical-eda-template)
35. [Mini EDA Project Example](#35-mini-eda-project-example)
36. [Best Practices](#36-best-practices)
37. [Key Takeaways](#37-key-takeaways)

---

# 1. Introduction to EDA

**Exploratory Data Analysis (EDA)** is the process of analyzing, summarizing, visualizing, and understanding a dataset before applying statistical models or machine learning algorithms.

EDA helps us answer questions such as:

* What does the dataset contain?
* How many rows and columns are present?
* What are the data types?
* Are there missing values?
* Are there duplicate records?
* Are there outliers?
* How are variables distributed?
* Which variables are related?
* Which features may be useful for prediction?
* Is the dataset suitable for machine learning?

### Simple Definition

> EDA is the process of understanding your data before building a model.

---

# 2. Why EDA is Important

EDA is one of the most important steps in a Data Science workflow.

Without EDA, a machine learning model may produce misleading results because of:

* Missing values
* Incorrect data types
* Duplicate records
* Outliers
* Incorrect assumptions
* Data leakage
* Highly correlated features
* Imbalanced classes
* Incorrect target values

### Benefits of EDA

1. Understand the dataset
2. Identify data-quality problems
3. Detect patterns
4. Find relationships between variables
5. Detect outliers
6. Understand distributions
7. Select useful features
8. Identify unnecessary features
9. Prepare data for preprocessing
10. Improve machine learning performance

---

# 3. EDA Workflow

A typical EDA workflow looks like this:

```text
                 Dataset
                    |
                    v
            Load the Dataset
                    |
                    v
          Understand Structure
                    |
                    v
          Check Data Types
                    |
                    v
          Check Missing Values
                    |
                    v
         Check Duplicate Values
                    |
                    v
       Statistical Analysis
                    |
                    v
        Univariate Analysis
                    |
                    v
         Bivariate Analysis
                    |
                    v
        Multivariate Analysis
                    |
                    v
         Outlier Detection
                    |
                    v
       Correlation Analysis
                    |
                    v
          Data Visualization
                    |
                    v
        Feature Engineering
                    |
                    v
       Machine Learning Ready
```

---

# 4. Types of Data

Understanding data types is essential during EDA.

## 4.1 Numerical Data

Numerical data represents numbers.

Examples:

```text
Age
Salary
Height
Weight
Experience
Price
Temperature
```

Numerical data can be divided into:

### Continuous Data

Can take decimal values.

Examples:

```text
Height = 172.5
Temperature = 36.7
Weight = 68.4
```

### Discrete Data

Usually represents countable values.

Examples:

```text
Number of students = 50
Number of orders = 120
Number of products = 10
```

---

# 4.2 Categorical Data

Categorical data represents categories.

Examples:

```text
Gender
City
Department
Product Category
Payment Method
```

Example:

```text
Gender:
Male
Female

Department:
IT
HR
Finance
Sales
```

---

# 4.3 Ordinal Data

Categories that have a meaningful order.

Example:

```text
Education:
School < Bachelor < Master < PhD
```

Another example:

```text
Satisfaction:
Poor < Average < Good < Excellent
```

---

# 4.4 Nominal Data

Categories without an inherent order.

Example:

```text
Color:
Red
Blue
Green
Black
```

There is no natural ranking between these categories.

---

# 5. Understanding the Dataset

Before performing detailed analysis, understand the dataset.

Important questions:

* What is the dataset about?
* What does each row represent?
* What does each column represent?
* What is the target variable?
* Which columns are numerical?
* Which columns are categorical?
* Are there identifiers?
* Are there timestamps?
* Are there missing values?

---

# 6. Loading Data with Pandas

Pandas is one of the most commonly used Python libraries for EDA.

```python
import pandas as pd
import numpy as np
```

### Load CSV

```python
df = pd.read_csv("data.csv")
```

### Load Excel

```python
df = pd.read_excel("data.xlsx")
```

### Load JSON

```python
df = pd.read_json("data.json")
```

### Load SQL Data

```python
import pandas as pd
import sqlite3

connection = sqlite3.connect("database.db")

df = pd.read_sql("SELECT * FROM employees", connection)
```

---

# 7. Initial Data Inspection

## View First Rows

```python
df.head()
```

View first 10 rows:

```python
df.head(10)
```

---

## View Last Rows

```python
df.tail()
```

---

## Random Samples

```python
df.sample(5)
```

---

## Dataset Shape

```python
df.shape
```

Example:

```text
(1000, 12)
```

This means:

```text
Rows    = 1000
Columns = 12
```

---

## Column Names

```python
df.columns
```

---

## Dataset Information

```python
df.info()
```

This provides:

* Column names
* Number of non-null values
* Data types
* Memory usage

---

# 8. Data Types

Check data types:

```python
df.dtypes
```

Example:

```text
Age          int64
Salary       float64
Department   object
JoiningDate  datetime64
```

### Convert Data Types

```python
df["Age"] = df["Age"].astype(int)
```

Convert to categorical:

```python
df["Department"] = df["Department"].astype("category")
```

Convert to datetime:

```python
df["JoiningDate"] = pd.to_datetime(df["JoiningDate"])
```

---

# 9. Missing Values

Missing values are one of the most common problems in real-world datasets.

## Check Missing Values

```python
df.isnull()
```

Count missing values:

```python
df.isnull().sum()
```

Percentage of missing values:

```python
df.isnull().mean() * 100
```

---

## Visualize Missing Values

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df.isnull(), cbar=False)
plt.show()
```

---

# 9.1 Handling Missing Values

There are several strategies.

## Remove Rows

```python
df.dropna()
```

Remove rows based on a column:

```python
df.dropna(subset=["Age"])
```

---

## Fill with Mean

```python
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())
```

---

## Fill with Median

```python
df["Salary"] = df["Salary"].fillna(df["Salary"].median())
```

Median is often useful when the data contains outliers.

---

## Fill Categorical Values with Mode

```python
df["City"] = df["City"].fillna(df["City"].mode()[0])
```

---

## Forward Fill

```python
df.fillna(method="ffill")
```

---

## Backward Fill

```python
df.fillna(method="bfill")
```

---

# 10. Duplicate Values

Duplicates can negatively affect analysis and machine learning.

## Check Duplicates

```python
df.duplicated().sum()
```

---

## Display Duplicate Rows

```python
df[df.duplicated()]
```

---

## Remove Duplicates

```python
df.drop_duplicates(inplace=True)
```

---

## Duplicate Based on Specific Columns

```python
df.duplicated(subset=["Email"]).sum()
```

Remove:

```python
df.drop_duplicates(subset=["Email"], inplace=True)
```

---

# 11. Statistical Summary

Use:

```python
df.describe()
```

This provides:

* Count
* Mean
* Standard deviation
* Minimum
* 25th percentile
* Median
* 75th percentile
* Maximum

Example:

```text
Age

count     1000
mean        29.5
std          5.8
min         18
25%         25
50%         29
75%         34
max         55
```

---

## Categorical Summary

```python
df.describe(include="object")
```

All columns:

```python
df.describe(include="all")
```

---

# 12. Univariate Analysis

**Univariate analysis** analyzes one variable at a time.

Examples:

```text
Age
Salary
Gender
Department
Price
```

Questions:

* What is the distribution?
* What is the average?
* What is the most common value?
* Are there outliers?

---

# 12.1 Numerical Univariate Analysis

```python
df["Age"].describe()
```

Histogram:

```python
df["Age"].hist()
plt.show()
```

---

# 12.2 Categorical Univariate Analysis

Count values:

```python
df["Department"].value_counts()
```

Percentage:

```python
df["Department"].value_counts(normalize=True) * 100
```

Bar chart:

```python
df["Department"].value_counts().plot(kind="bar")
plt.show()
```

---

# 13. Bivariate Analysis

Bivariate analysis studies the relationship between **two variables**.

Examples:

```text
Age vs Salary
Experience vs Salary
Gender vs Salary
Department vs Salary
```

---

## Numerical vs Numerical

Example:

```python
sns.scatterplot(data=df, x="Experience", y="Salary")
plt.show()
```

This can help determine whether salary increases with experience.

---

## Categorical vs Numerical

Example:

```python
sns.boxplot(data=df, x="Department", y="Salary")
plt.show()
```

This compares salary distributions across departments.

---

## Categorical vs Categorical

Example:

```python
pd.crosstab(df["Gender"], df["Department"])
```

Visualization:

```python
pd.crosstab(
    df["Gender"],
    df["Department"]
).plot(kind="bar")
```

---

# 14. Multivariate Analysis

Multivariate analysis studies relationships between more than two variables.

Example:

```text
Age
Experience
Education
Department
Salary
```

One common technique is a pair plot.

```python
sns.pairplot(df)
plt.show()
```

For selected columns:

```python
sns.pairplot(
    df[["Age", "Experience", "Salary"]]
)
plt.show()
```

---

# 15. Categorical Data Analysis

Categorical variables can be analyzed using:

```python
value_counts()
```

Example:

```python
df["City"].value_counts()
```

Top 5 categories:

```python
df["City"].value_counts().head(5)
```

---

## Cross Tabulation

```python
pd.crosstab(
    df["Gender"],
    df["Department"]
)
```

---

## Category Percentages

```python
df["Department"].value_counts(normalize=True) * 100
```

---

# 16. Numerical Data Analysis

Common numerical statistics:

```python
df["Salary"].mean()
```

```python
df["Salary"].median()
```

```python
df["Salary"].mode()
```

```python
df["Salary"].std()
```

```python
df["Salary"].min()
```

```python
df["Salary"].max()
```

---

# 16.1 Mean

Mean is the arithmetic average.

Formula:

```text
Mean = Sum of all values / Number of values
```

Python:

```python
df["Salary"].mean()
```

---

# 16.2 Median

Median is the middle value after sorting.

```python
df["Salary"].median()
```

Median is less affected by extreme outliers than mean.

---

# 16.3 Mode

Mode is the most frequently occurring value.

```python
df["Department"].mode()
```

---

# 17. Outlier Detection

An outlier is an observation that is significantly different from other observations.

Example:

```text
Salary:

30,000
35,000
40,000
42,000
45,000
5,00,000
```

`5,00,000` may be an outlier.

---

# 17.1 Detect Outliers Using Boxplot

```python
sns.boxplot(x=df["Salary"])
plt.show()
```

---

# 17.2 IQR Method

IQR means:

```text
IQR = Q3 - Q1
```

Where:

```text
Q1 = 25th percentile
Q3 = 75th percentile
```

Lower boundary:

```text
Q1 - 1.5 × IQR
```

Upper boundary:

```text
Q3 + 1.5 × IQR
```

Python:

```python
Q1 = df["Salary"].quantile(0.25)
Q3 = df["Salary"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[
    (df["Salary"] < lower) |
    (df["Salary"] > upper)
]
```

---

# 17.3 Z-Score Method

Z-score measures how many standard deviations a value is away from the mean.

Formula:

```text
Z = (X - μ) / σ
```

Python:

```python
from scipy.stats import zscore

df["z_score"] = zscore(df["Salary"])
```

Common rule:

```text
Z > +3
Z < -3
```

may indicate potential outliers.

---

# 18. Correlation Analysis

Correlation measures the strength and direction of the relationship between numerical variables.

Correlation ranges from:

```text
-1 to +1
```

### Interpretation

```text
+1  = Perfect positive correlation
 0  = No linear correlation
-1  = Perfect negative correlation
```

---

## Positive Correlation

When one variable increases, another tends to increase.

Example:

```text
Experience ↑
Salary ↑
```

---

## Negative Correlation

When one variable increases, another tends to decrease.

Example:

```text
Price ↑
Demand ↓
```

---

## Calculate Correlation

```python
df.corr(numeric_only=True)
```

---

## Correlation Heatmap

```python
plt.figure(figsize=(10, 6))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.show()
```

---

# 19. Covariance

Covariance indicates how two variables change together.

```python
df[["Age", "Salary"]].cov()
```

Difference between correlation and covariance:

| Covariance                          | Correlation                      |
| ----------------------------------- | -------------------------------- |
| Indicates direction of relationship | Indicates direction and strength |
| Scale-dependent                     | Scale-independent                |
| Range is not fixed                  | Range is -1 to +1                |

---

# 20. Distribution Analysis

Distribution describes how values are spread.

Common distributions include:

* Normal distribution
* Uniform distribution
* Binomial distribution
* Poisson distribution
* Exponential distribution
* Right-skewed distribution
* Left-skewed distribution

---

## Histogram

```python
sns.histplot(df["Salary"], kde=True)
plt.show()
```

Histogram helps identify:

* Center
* Spread
* Shape
* Skewness
* Potential outliers

---

# 21. Skewness

Skewness measures the asymmetry of a distribution.

```python
df["Salary"].skew()
```

### Interpretation

```text
Skewness ≈ 0
```

Approximately symmetric.

```text
Skewness > 0
```

Right-skewed.

```text
Skewness < 0
```

Left-skewed.

---

## Right-Skewed Distribution

The tail extends toward larger values.

Example:

```text
Income
```

A small number of people may have extremely high incomes.

---

# 22. Kurtosis

Kurtosis describes the shape of the tails of a distribution.

```python
df["Salary"].kurt()
```

It can help identify distributions with unusually heavy or light tails.

High kurtosis generally indicates heavier tails and more extreme values.

---

# 23. Feature Relationships

EDA should identify relationships between features.

For example:

```text
Experience → Salary
Age → Salary
Education → Salary
Department → Salary
```

Useful visualizations:

* Scatter plots
* Box plots
* Violin plots
* Heatmaps
* Pair plots
* Bar charts

---

# 24. GroupBy Analysis

`groupby()` is one of the most useful Pandas functions for EDA.

Example:

```python
df.groupby("Department")["Salary"].mean()
```

---

## Multiple Aggregations

```python
df.groupby("Department")["Salary"].agg(
    ["mean", "median", "min", "max"]
)
```

---

## Multiple Columns

```python
df.groupby("Department").agg({
    "Salary": "mean",
    "Age": "mean",
    "Experience": "mean"
})
```

---

# 25. Pivot Tables

Pivot tables summarize data across multiple dimensions.

```python
pd.pivot_table(
    df,
    values="Salary",
    index="Department",
    columns="Gender",
    aggfunc="mean"
)
```

Pivot tables are useful for:

* Business analysis
* Reporting
* Comparing groups
* Finding trends

---

# 26. Visualization for EDA

Visualization makes patterns easier to understand.

Popular libraries:

```python
import matplotlib.pyplot as plt
import seaborn as sns
```

Common visualization types:

* Bar chart
* Histogram
* Box plot
* Violin plot
* Scatter plot
* Line chart
* Pie chart
* Heatmap
* Pair plot

---

# 27. Common EDA Plots

## 27.1 Bar Chart

Useful for categorical variables.

```python
sns.countplot(data=df, x="Department")
plt.show()
```

---

## 27.2 Histogram

Useful for numerical distributions.

```python
sns.histplot(data=df, x="Salary", kde=True)
plt.show()
```

---

## 27.3 Box Plot

Useful for:

* Outliers
* Median
* Quartiles
* Distribution comparison

```python
sns.boxplot(data=df, x="Salary")
plt.show()
```

---

## 27.4 Violin Plot

Combines aspects of a box plot and density plot.

```python
sns.violinplot(
    data=df,
    x="Department",
    y="Salary"
)

plt.show()
```

---

## 27.5 Scatter Plot

Used to examine relationships between numerical variables.

```python
sns.scatterplot(
    data=df,
    x="Experience",
    y="Salary"
)

plt.show()
```

---

## 27.6 Line Plot

Useful for time-series data.

```python
sns.lineplot(
    data=df,
    x="Date",
    y="Sales"
)

plt.show()
```

---

## 27.7 Heatmap

Useful for correlation matrices.

```python
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True
)

plt.show()
```

---

## 27.8 Pair Plot

```python
sns.pairplot(df)
plt.show()
```

Useful for understanding relationships among multiple numerical variables.

---

# 28. Data Quality Checks

A good EDA process should include a systematic data-quality check.

### Check Number of Rows

```python
df.shape[0]
```

### Check Number of Columns

```python
df.shape[1]
```

### Check Missing Values

```python
df.isnull().sum()
```

### Check Duplicates

```python
df.duplicated().sum()
```

### Check Data Types

```python
df.dtypes
```

### Check Unique Values

```python
df.nunique()
```

### Check Negative Values

```python
(df["Salary"] < 0).sum()
```

### Check Infinite Values

```python
import numpy as np

np.isinf(df.select_dtypes(include=np.number)).sum()
```

---

# 29. Feature Engineering During EDA

Feature engineering means creating new useful features from existing data.

Example:

```text
Date → Year
Date → Month
Date → Day
Date → Day of Week
```

---

## Extract Date Features

```python
df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek
```

---

## Create Age Groups

```python
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 18, 30, 45, 60, 100],
    labels=[
        "Child",
        "Young Adult",
        "Adult",
        "Middle Age",
        "Senior"
    ]
)
```

---

# 30. Target Variable Analysis

In supervised machine learning, the **target variable** is the variable we want to predict.

Examples:

```text
House Price
Disease
Salary
Customer Churn
Loan Approval
```

---

## Classification Target

Example:

```text
Disease:
Healthy
Cataract
Glaucoma
```

Check distribution:

```python
df["Disease"].value_counts()
```

Percentage:

```python
df["Disease"].value_counts(normalize=True) * 100
```

This helps identify **class imbalance**.

---

## Regression Target

For a numerical target:

```python
df["Price"].describe()
```

Visualize:

```python
sns.histplot(df["Price"], kde=True)
plt.show()
```

---

# 31. EDA for Machine Learning

EDA should prepare us for machine learning.

Typical process:

```text
Raw Dataset
     |
     v
Data Understanding
     |
     v
Data Cleaning
     |
     v
Missing Value Analysis
     |
     v
Duplicate Detection
     |
     v
Outlier Analysis
     |
     v
Distribution Analysis
     |
     v
Correlation Analysis
     |
     v
Feature Analysis
     |
     v
Feature Engineering
     |
     v
Encoding
     |
     v
Scaling
     |
     v
Train/Test Split
     |
     v
Machine Learning
```

---

# 31.1 Important ML Consideration: Data Leakage

**Data leakage** occurs when information that should not be available during training is used by the model.

Example:

```text
Target = Loan Approved
```

If a feature contains information that was generated after loan approval, using it can cause leakage.

EDA helps identify suspicious columns.

---

# 31.2 Feature Selection

EDA can help identify:

* Highly correlated features
* Irrelevant features
* Constant features
* Duplicate features
* Features with excessive missing values
* Features with data leakage

---

# 32. EDA Checklist

Use this checklist for every dataset.

## Dataset Understanding

* [ ] Understand the problem statement
* [ ] Understand each column
* [ ] Identify target variable
* [ ] Identify feature variables
* [ ] Identify numerical columns
* [ ] Identify categorical columns
* [ ] Identify date/time columns

## Data Inspection

* [ ] Load dataset
* [ ] Check shape
* [ ] Check column names
* [ ] Check first rows
* [ ] Check last rows
* [ ] Check data types
* [ ] Check memory usage

## Data Quality

* [ ] Check missing values
* [ ] Check duplicates
* [ ] Check invalid values
* [ ] Check inconsistent categories
* [ ] Check impossible values
* [ ] Check infinite values

## Statistical Analysis

* [ ] Mean
* [ ] Median
* [ ] Mode
* [ ] Standard deviation
* [ ] Minimum
* [ ] Maximum
* [ ] Percentiles
* [ ] Skewness
* [ ] Kurtosis

## Visualization

* [ ] Histogram
* [ ] Bar chart
* [ ] Box plot
* [ ] Scatter plot
* [ ] Correlation heatmap
* [ ] Pair plot
* [ ] Line chart if time-series data exists

## Feature Analysis

* [ ] Check correlations
* [ ] Detect outliers
* [ ] Analyze distributions
* [ ] Identify useful features
* [ ] Identify irrelevant features
* [ ] Analyze target variable

---

# 33. Common EDA Mistakes

## Mistake 1: Starting ML Without EDA

Never directly train a model without understanding the dataset.

---

## Mistake 2: Ignoring Missing Values

Missing values can significantly affect model performance.

---

## Mistake 3: Removing All Outliers

Not every outlier is an error.

Some outliers represent genuine observations.

---

## Mistake 4: Using Only Mean

Mean can be heavily affected by outliers.

Always consider:

```text
Mean
Median
Distribution
```

---

## Mistake 5: Ignoring Class Imbalance

Example:

```text
Class A = 95%
Class B = 5%
```

Accuracy alone may become misleading.

---

## Mistake 6: Ignoring Data Types

A numeric-looking column may actually contain strings.

Example:

```text
"₹50,000"
"₹60,000"
```

This needs cleaning before numerical analysis.

---

## Mistake 7: Too Many Visualizations Without Purpose

EDA is not about generating hundreds of graphs.

Every visualization should answer a question.

---

# 34. Practical EDA Template

The following template can be reused for most CSV datasets.

```python
# ==========================================
# Exploratory Data Analysis Template
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------
# 1. Load Dataset
# ------------------------------------------

df = pd.read_csv("data.csv")

# ------------------------------------------
# 2. Basic Information
# ------------------------------------------

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nDataset Info:")
df.info()

# ------------------------------------------
# 3. Preview
# ------------------------------------------

print("\nFirst 5 Rows:")
print(df.head())

print("\nRandom Sample:")
print(df.sample(5))

# ------------------------------------------
# 4. Missing Values
# ------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

print("\nMissing Percentage:")
print(df.isnull().mean() * 100)

# ------------------------------------------
# 5. Duplicate Values
# ------------------------------------------

print("\nDuplicates:")
print(df.duplicated().sum())

# ------------------------------------------
# 6. Statistical Summary
# ------------------------------------------

print("\nStatistical Summary:")
print(df.describe())

# ------------------------------------------
# 7. Categorical Summary
# ------------------------------------------

print("\nCategorical Summary:")
print(df.describe(include="object"))

# ------------------------------------------
# 8. Unique Values
# ------------------------------------------

print("\nUnique Values:")
print(df.nunique())

# ------------------------------------------
# 9. Correlation
# ------------------------------------------

correlation = df.corr(numeric_only=True)

print("\nCorrelation:")
print(correlation)

# ------------------------------------------
# 10. Correlation Heatmap
# ------------------------------------------

plt.figure(figsize=(10, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

# ------------------------------------------
# 11. Numerical Columns
# ------------------------------------------

numeric_columns = df.select_dtypes(
    include=np.number
).columns

print("\nNumerical Columns:")
print(numeric_columns)

# ------------------------------------------
# 12. Categorical Columns
# ------------------------------------------

categorical_columns = df.select_dtypes(
    include="object"
).columns

print("\nCategorical Columns:")
print(categorical_columns)
```

---

# 35. Mini EDA Project Example

Consider an employee dataset:

```text
EmployeeID
Name
Age
Gender
Department
Experience
Salary
City
```

## Step 1 — Load Dataset

```python
df = pd.read_csv("employees.csv")
```

---

## Step 2 — Understand Structure

```python
df.shape
df.info()
df.head()
```

---

## Step 3 — Missing Values

```python
df.isnull().sum()
```

---

## Step 4 — Duplicate Employees

```python
df.duplicated().sum()
```

---

## Step 5 — Salary Statistics

```python
df["Salary"].describe()
```

---

## Step 6 — Department Distribution

```python
df["Department"].value_counts()
```

---

## Step 7 — Average Salary by Department

```python
df.groupby("Department")["Salary"].mean()
```

---

## Step 8 — Experience vs Salary

```python
sns.scatterplot(
    data=df,
    x="Experience",
    y="Salary"
)

plt.show()
```

---

## Step 9 — Salary Distribution

```python
sns.histplot(
    data=df,
    x="Salary",
    kde=True
)

plt.show()
```

---

## Step 10 — Salary Outliers

```python
sns.boxplot(
    data=df,
    x="Salary"
)

plt.show()
```

---

## Step 11 — Correlation

```python
df.corr(numeric_only=True)
```

---

## Step 12 — Insights

After completing EDA, write conclusions such as:

```text
1. The dataset contains 1000 employee records.
2. The IT department has the highest number of employees.
3. Salary has a positive relationship with experience.
4. Several salary values appear to be outliers.
5. Age and experience show moderate correlation.
6. Some columns contain missing values.
7. The salary distribution is right-skewed.
```

---

# 36. Best Practices

### 1. Understand the Business Problem First

Do not analyze data blindly.

Understand:

```text
What problem are we solving?
What decision will the analysis support?
What are we trying to predict?
```

---

### 2. Always Start with Data Profiling

Check:

```python
df.shape
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
```

---

### 3. Visualize Important Variables

Use visualizations to identify:

* Trends
* Patterns
* Outliers
* Relationships
* Distributions

---

### 4. Compare Mean and Median

If:

```text
Mean ≈ Median
```

the distribution may be relatively symmetric.

If:

```text
Mean >> Median
```

the data may be right-skewed.

If:

```text
Mean << Median
```

the data may be left-skewed.

---

### 5. Don't Automatically Remove Outliers

First determine:

```text
Is it a data-entry error?
Is it a genuine observation?
Is it important to the business problem?
```

---

### 6. Document Your Findings

An EDA notebook should not contain only code.

Add explanations:

```text
Observation:
Salary is strongly associated with experience.

Conclusion:
Experience may be an important predictive feature.
```

---

# 37. Key Takeaways

EDA is the foundation of Data Science.

Remember the following workflow:

```text
Understand
    ↓
Inspect
    ↓
Clean
    ↓
Summarize
    ↓
Visualize
    ↓
Analyze
    ↓
Find Patterns
    ↓
Engineer Features
    ↓
Prepare for ML
```

### Most Important Pandas Commands

```python
df.head()
df.tail()
df.shape
df.info()
df.describe()
df.dtypes
df.columns
df.nunique()
df.isnull().sum()
df.duplicated().sum()
df.value_counts()
df.groupby()
df.corr()
```

### Most Important Visualization Techniques

```text
Histogram
Box Plot
Bar Chart
Scatter Plot
Line Plot
Violin Plot
Heatmap
Pair Plot
```

### Most Important EDA Concepts

```text
Missing Values
Duplicates
Outliers
Distribution
Mean
Median
Mode
Variance
Standard Deviation
Skewness
Kurtosis
Correlation
Covariance
Feature Relationships
Class Imbalance
Data Leakage
Feature Engineering
```

---

# EDA Quick Revision

```text
EDA
│
├── Data Understanding
│   ├── Shape
│   ├── Columns
│   ├── Data Types
│   └── Target Variable
│
├── Data Quality
│   ├── Missing Values
│   ├── Duplicates
│   ├── Invalid Values
│   └── Inconsistent Values
│
├── Statistics
│   ├── Mean
│   ├── Median
│   ├── Mode
│   ├── Standard Deviation
│   ├── Percentiles
│   ├── Skewness
│   └── Kurtosis
│
├── Analysis
│   ├── Univariate
│   ├── Bivariate
│   └── Multivariate
│
├── Visualization
│   ├── Histogram
│   ├── Box Plot
│   ├── Bar Chart
│   ├── Scatter Plot
│   ├── Heatmap
│   └── Pair Plot
│
├── Advanced Analysis
│   ├── Correlation
│   ├── Covariance
│   ├── Outliers
│   ├── GroupBy
│   └── Pivot Tables
│
└── ML Preparation
    ├── Feature Selection
    ├── Feature Engineering
    ├── Data Leakage Check
    ├── Encoding
    └── Scaling
```

> **Final Rule:**
> **Never build a machine learning model before understanding your data. EDA turns raw data into meaningful information and helps you make better modeling decisions.**
