# 🌳 Decision Tree in Machine Learning

> A complete learning resource covering Decision Tree fundamentals,
> algorithms, splitting criteria, pruning, implementation, evaluation,
> advanced concepts, best practices, interview questions, and a
> practical mini-project.

------------------------------------------------------------------------

## 📚 Table of Contents

1.  [🌳 Introduction](#1--introduction)
2.  [🎯 What is a Decision Tree?](#2--what-is-a-decision-tree)
3.  [🧩 Decision Tree Terminology](#3--decision-tree-terminology)
4.  [🔄 How a Decision Tree Works](#4--how-a-decision-tree-works)
5.  [🌿 Structure of a Decision Tree](#5--structure-of-a-decision-tree)
6.  [📊 Classification vs Regression
    Trees](#6--classification-vs-regression-trees)
7.  [✂️ Splitting and Decision Rules](#7--splitting-and-decision-rules)
8.  [📐 Splitting Criteria](#8--splitting-criteria)
9.  [🟢 Gini Impurity](#9--gini-impurity)
10. [📈 Entropy and Information Gain](#10--entropy-and-information-gain)
11. [⚖️ Gini vs Entropy](#11--gini-vs-entropy)
12. [📉 Decision Trees for
    Regression](#12--decision-trees-for-regression)
13. [✂️ Pruning](#13--pruning)
14. [🎛️ Important Hyperparameters](#14--important-hyperparameters)
15. [💻 Decision Tree with
    Scikit-Learn](#15--decision-tree-with-scikit-learn)
16. [🧪 Practical Classification
    Example](#16--practical-classification-example)
17. [📊 Model Evaluation](#17--model-evaluation)
18. [🌍 Real-World Use Cases](#18--real-world-use-cases)
19. [✅ Advantages](#19--advantages)
20. [⚠️ Limitations](#20--limitations)
21. [🚫 Common Mistakes](#21--common-mistakes)
22. [🧠 Best Practices](#22--best-practices)
23. [🚀 Advanced Concepts](#23--advanced-concepts)
24. [🌲 Ensemble Methods Based on
    Trees](#24--ensemble-methods-based-on-trees)
25. [🔍 Feature Importance](#25--feature-importance)
26. [🧩 Handling Missing and Categorical
    Data](#26--handling-missing-and-categorical-data)
27. [🛠️ Practical Mini-Project](#27--practical-mini-project)
28. [🎤 Interview Questions and
    Points](#28--interview-questions-and-points)
29. [📝 Quick Revision](#29--quick-revision)
30. [🗺️ Visual Roadmap](#30--visual-roadmap)

------------------------------------------------------------------------

# 1. 🌳 Introduction

Decision Tree is one of the most intuitive and widely used supervised
machine learning algorithms.

It can be used for:

-   Classification
-   Regression
-   Feature selection
-   Rule extraction
-   Exploratory analysis

The core idea is simple:

> **Repeatedly split the dataset into smaller groups using the feature
> and threshold that produce the best separation of the target values.**

A Decision Tree resembles a flowchart:

``` text
                    Is Age <= 30?
                    /           \
                  Yes            No
                  /               \
          Is Income > 50K?       Approve
             /       \
           Yes        No
           /           \
       Approve       Reject
```

------------------------------------------------------------------------

# 2. 🎯 What is a Decision Tree?

A **Decision Tree** is a supervised learning algorithm that makes
predictions by learning a sequence of decision rules from training data.

It consists of:

-   Root node
-   Internal decision nodes
-   Branches
-   Leaf nodes

For classification, a leaf predicts a class.

For regression, a leaf predicts a numerical value.

### Basic idea

Suppose we want to predict whether a person will buy a product.

Features:

-   Age
-   Income
-   Previous purchases
-   Website visits

The tree may learn:

``` text
                    Age < 30?
                   /         \
                 Yes          No
                 /             \
        Visits > 5?          Income > 70K?
          /    \               /      \
        Yes    No            Yes       No
        Buy   Don't Buy      Buy      Don't Buy
```

The model has converted numerical data into a sequence of understandable
rules.

------------------------------------------------------------------------

# 3. 🧩 Decision Tree Terminology

  Term            Meaning
  --------------- -----------------------------------------
  Root Node       First/top decision node
  Internal Node   Node where a decision/split occurs
  Branch          Connection between two nodes
  Leaf Node       Final prediction node
  Parent Node     Node that is split
  Child Node      Resulting node after a split
  Depth           Number of levels in the tree
  Split           Division of data based on a condition
  Impurity        Degree of mixed target classes
  Criterion       Metric used to evaluate a split
  Pruning         Removing unnecessary branches
  Threshold       Value used to split a numerical feature

### Example

``` text
                 Root
              Age < 30?
             /         \
        Child           Child
       Income > 50K      ...
       /       \
     Leaf      Leaf
```

------------------------------------------------------------------------

# 4. 🔄 How a Decision Tree Works

The general process is:

1.  Start with the complete training dataset.
2.  Examine candidate features.
3.  Generate possible split points.
4.  Calculate the quality of each split.
5.  Select the best split.
6.  Divide the dataset.
7.  Repeat the process recursively.
8.  Stop according to stopping conditions.
9.  Assign predictions to leaf nodes.
10. Optionally prune the tree.

## 🔁 Training Workflow

``` mermaid
flowchart TD
    A[Training Dataset] --> B[Select Candidate Features]
    B --> C[Generate Possible Splits]
    C --> D[Calculate Split Criterion]
    D --> E{Best Split?}
    E -->|Yes| F[Split Dataset]
    E -->|No| G[Create Leaf]
    F --> H{Stopping Condition?}
    H -->|No| B
    H -->|Yes| G
    G --> I[Build Decision Tree]
    I --> J[Optional Pruning]
    J --> K[Final Model]
```

------------------------------------------------------------------------

# 5. 🌿 Structure of a Decision Tree

A Decision Tree can be represented as:

``` text
                         ROOT
                           |
                    Feature A <= 10?
                    /              \
                  Yes               No
                  /                  \
             Feature B > 5?       LEAF: Class B
              /       \
            Yes       No
            /          \
      LEAF: Class A   LEAF: Class B
```

## 🌱 Root Node

The root is the first split.

It is usually selected because it provides the highest improvement in
the chosen splitting criterion.

## 🌿 Internal Node

An internal node represents a decision condition.

Example:

``` text
Age <= 35
```

## 🍃 Leaf Node

A leaf contains the final prediction.

Example:

``` text
Prediction = Approved
```

------------------------------------------------------------------------

# 6. 📊 Classification vs Regression Trees

  -----------------------------------------------------------------------
  Property                Classification Tree     Regression Tree
  ----------------------- ----------------------- -----------------------
  Target                  Categorical             Numerical

  Output                  Class                   Number

  Example                 Spam / Not Spam         House Price

  Common Criterion        Gini, Entropy           MSE, MAE

  Leaf Prediction         Majority                Mean/median depending
                          class/probability       on criterion

  Evaluation              Accuracy, Precision,    MAE, MSE, RMSE, R²
                          Recall, F1              
  -----------------------------------------------------------------------

### Classification

``` text
Input → Decision Rules → Class
```

Example:

``` text
Email → Tree → Spam
```

### Regression

``` text
Input → Decision Rules → Numeric Value
```

Example:

``` text
House Features → Tree → ₹85,00,000
```

------------------------------------------------------------------------

# 7. ✂️ Splitting and Decision Rules

A split divides the data into subsets.

For a numerical feature:

``` text
Age <= 30
```

creates:

``` text
Left Child:
Age <= 30

Right Child:
Age > 30
```

For a categorical feature:

``` text
City = Pune
```

can divide observations based on category values, depending on the
implementation.

## 🎯 Goal of Splitting

The objective is to create child nodes that are more homogeneous than
their parent node.

For classification:

> A good split creates groups containing mostly one class.

For regression:

> A good split creates groups with similar target values.

------------------------------------------------------------------------

# 8. 📐 Splitting Criteria

Different Decision Tree algorithms use different measures.

  Criterion          Mainly Used For   Main Idea
  ------------------ ----------------- -----------------------------
  Gini Impurity      Classification    Measures class impurity
  Entropy            Classification    Measures uncertainty
  Information Gain   Classification    Reduction in entropy
  MSE                Regression        Reduction in squared error
  MAE                Regression        Reduction in absolute error

The algorithm searches for a split that improves node purity.

------------------------------------------------------------------------

# 9. 🟢 Gini Impurity

Gini impurity measures how mixed the classes are inside a node.

## 📐 Formula

$$
Gini = 1 - \sum_{i=1}^{C} p_i^2
$$

Where:

-   $C$ = number of classes
-   $p_i$ = proportion of observations belonging to class $i$

## Example

Suppose a node contains:

-   6 Yes
-   4 No

Then:

$$
p(Yes)=0.6
$$

$$
p(No)=0.4
$$

Therefore:

$$
Gini = 1-(0.6^2+0.4^2)
$$

$$
Gini = 1-(0.36+0.16)
$$

$$
Gini = 0.48
$$

### Interpretation

          Gini Interpretation
  ------------ -----------------
             0 Completely pure
    Close to 0 Mostly pure
        Higher More mixed

For a binary classification problem, the maximum Gini impurity is 0.5.

------------------------------------------------------------------------

# 10. 📈 Entropy and Information Gain

## 🔐 Entropy

Entropy measures uncertainty or disorder.

$$
Entropy = -\sum_{i=1}^{C}p_i\log_2(p_i)
$$

### Interpretation

    Entropy Meaning
  --------- ------------------
          0 Pure node
        Low Mostly one class
       High Highly mixed

For binary classification, maximum entropy is 1 when both classes are
equally represented.

------------------------------------------------------------------------

## 📉 Information Gain

Information Gain measures the reduction in entropy after a split.

$$
IG = Entropy(parent) -
\sum_{j=1}^{k}
\frac{N_j}{N}
Entropy(child_j)
$$

Where:

-   $N$ = number of samples in parent
-   $N_j$ = samples in child $j$

### Simple idea

``` text
Before Split
     ↓
High Uncertainty
     ↓
     Split
     ↓
Lower Uncertainty
```

A split with higher Information Gain is generally preferred by ID3-style
tree learning.

------------------------------------------------------------------------

# 11. ⚖️ Gini vs Entropy

  Feature                    Gini                  Entropy
  -------------------------- --------------------- -----------------------
  Measures                   Impurity              Uncertainty
  Formula                    $1-\sum p_i^2$        $-\sum p_i\log_2 p_i$
  Range                      0 to 0.5 for binary   0 to 1 for binary
  Computational Cost         Lower                 Slightly higher
  Common sklearn criterion   `gini`                `entropy`, `log_loss`
  Interpretation             Simpler               Information-theoretic

### Which should you use?

In many practical datasets, Gini and Entropy produce similar trees.

Start with:

``` python
criterion="gini"
```

and compare using cross-validation if model performance matters.

------------------------------------------------------------------------

# 12. 📉 Decision Trees for Regression

A Regression Tree predicts a continuous value.

Example:

``` text
                 Area <= 1500?
                  /          \
                Yes           No
                /              \
          Rooms <= 2?         ₹90L
           /     \
         ₹45L    ₹60L
```

## 📐 Mean Squared Error

A common regression criterion is:

$$
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y})^2
$$

The tree searches for splits that reduce the error within child nodes.

## Example

Target values:

``` text
[40, 42, 41, 90, 92, 88]
```

A good split may separate:

``` text
[40, 42, 41]
```

from:

``` text
[90, 92, 88]
```

because the values within each group are relatively similar.

------------------------------------------------------------------------

# 13. ✂️ Pruning

A fully grown tree can become extremely complex.

This may cause **overfitting**.

``` text
Small Tree
   ↓
May Underfit

Very Large Tree
   ↓
May Overfit

Optimized Tree
   ↓
Better Generalization
```

## 🌿 Pre-Pruning

Stop tree growth early using parameters such as:

-   `max_depth`
-   `min_samples_split`
-   `min_samples_leaf`
-   `max_leaf_nodes`
-   `max_features`

## ✂️ Post-Pruning

First grow the tree and then remove unnecessary branches.

One important method is **Cost-Complexity Pruning**.

### Cost-Complexity Objective

A simplified form is:

$$
R_\alpha(T)=R(T)+\alpha|T|
$$

Where:

-   $R(T)$ = tree error
-   $|T|$ = number of terminal nodes
-   $\alpha$ = complexity penalty

Larger $\alpha$ generally produces a simpler tree.

------------------------------------------------------------------------

# 14. 🎛️ Important Hyperparameters

  -----------------------------------------------------------------------
  Hyperparameter          Purpose                 Typical Effect
  ----------------------- ----------------------- -----------------------
  `criterion`             Split quality           Controls splitting
                                                  metric

  `max_depth`             Maximum depth           Limits complexity

  `min_samples_split`     Minimum samples         Prevents tiny splits
                          required to split       

  `min_samples_leaf`      Minimum samples per     Smooths predictions
                          leaf                    

  `max_leaf_nodes`        Maximum leaves          Controls tree size

  `max_features`          Features considered at  Adds randomness
                          split                   

  `ccp_alpha`             Pruning strength        Higher value → simpler
                                                  tree

  `random_state`          Reproducibility         Same results across
                                                  runs
  -----------------------------------------------------------------------

### Example

``` python
DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42
)
```

------------------------------------------------------------------------

# 15. 💻 Decision Tree with Scikit-Learn

## 📦 Import

``` python
from sklearn.tree import DecisionTreeClassifier
```

## 🏗️ Create Model

``` python
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)
```

## 🏋️ Train

``` python
model.fit(X_train, y_train)
```

## 🔮 Predict

``` python
y_pred = model.predict(X_test)
```

## 📊 Probability Prediction

``` python
y_prob = model.predict_proba(X_test)
```

### Complete basic workflow

``` python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
```

------------------------------------------------------------------------

# 16. 🧪 Practical Classification Example

Let's use the Breast Cancer Wisconsin dataset available through
Scikit-Learn.

## Step 1: Load Dataset

``` python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = data.data
y = data.target
```

## Step 2: Split Dataset

``` python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

## Step 3: Train Decision Tree

``` python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)
```

## Step 4: Predict

``` python
y_pred = model.predict(X_test)
```

## Step 5: Evaluate

``` python
from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))

print(classification_report(
    y_test,
    y_pred,
    target_names=data.target_names
))
```

### Why scaling is usually unnecessary

Decision Trees split data using conditions such as:

``` text
Feature <= Threshold
```

They are generally insensitive to monotonic feature scaling.

Therefore, unlike KNN, SVM, and many gradient-based models, feature
standardization is usually not required.

------------------------------------------------------------------------

# 17. 📊 Model Evaluation

## Classification Metrics

### Accuracy

$$
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}
$$

### Precision

$$
Precision = \frac{TP}{TP+FP}
$$

### Recall

$$
Recall = \frac{TP}{TP+FN}
$$

### F1 Score

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

### Confusion Matrix

``` python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
```

### Classification Report

``` python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

------------------------------------------------------------------------

## Regression Metrics

For a DecisionTreeRegressor, commonly used metrics include:

-   MAE
-   MSE
-   RMSE
-   R²

``` python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)
```

------------------------------------------------------------------------

# 18. 🌍 Real-World Use Cases

## 🏦 Banking

Predict whether a customer is likely to:

-   Default on a loan
-   Accept a loan
-   Require additional verification

## 🏥 Healthcare

Potential applications include:

-   Risk classification
-   Patient triage support
-   Disease classification

> Medical applications require appropriate validation, clinical
> oversight, and regulatory controls.

## 🛒 E-Commerce

Predict:

-   Customer purchase behavior
-   Customer churn
-   Product category

## 🏠 Real Estate

Predict:

-   House prices
-   Property categories
-   Rental demand

## 🎓 Education

Predict:

-   Student performance category
-   Dropout risk
-   Course recommendation

## 🚗 Automotive

Predict:

-   Vehicle maintenance needs
-   Customer purchase decisions
-   Component failure categories

------------------------------------------------------------------------

# 19. ✅ Advantages

  Advantage                      Explanation
  ------------------------------ -----------------------------------------
  Easy to Understand             Rules resemble human decision-making
  Easy to Visualize              Can be plotted as a tree
  Little Preprocessing           Scaling usually unnecessary
  Handles Non-Linearity          Captures nonlinear relationships
  Handles Feature Interactions   Naturally models interactions
  Feature Selection              Performs implicit feature selection
  Fast Prediction                Traverses only one path
  Flexible                       Works for classification and regression

------------------------------------------------------------------------

# 20. ⚠️ Limitations

  Limitation            Explanation
  --------------------- ------------------------------------------------------
  Overfitting           Deep trees can memorize training data
  Instability           Small data changes can produce different trees
  Greedy Optimization   Splits are usually selected locally
  Bias                  Some splitting approaches can favor certain features
  Poor Extrapolation    Regression trees do not extrapolate smoothly
  Large Trees           Can become difficult to interpret
  High Variance         Single trees can be unstable

### Key issue

> **A single Decision Tree often has high variance.**

This is one reason ensemble methods such as Random Forest and Gradient
Boosting are popular.

------------------------------------------------------------------------

# 21. 🚫 Common Mistakes

## ❌ Mistake 1: Growing the tree without constraints

``` python
model = DecisionTreeClassifier(random_state=42)
```

A completely unrestricted tree may overfit.

### Better

``` python
model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=3,
    random_state=42
)
```

------------------------------------------------------------------------

## ❌ Mistake 2: Evaluating only training accuracy

Example:

``` text
Training Accuracy = 100%
Testing Accuracy  = 82%
```

This can indicate overfitting.

Always evaluate on unseen data.

------------------------------------------------------------------------

## ❌ Mistake 3: Ignoring class imbalance

Accuracy may be misleading when one class dominates.

Use:

-   Precision
-   Recall
-   F1
-   ROC-AUC
-   PR-AUC
-   Confusion Matrix

------------------------------------------------------------------------

## ❌ Mistake 4: Assuming deeper always means better

Increasing depth generally increases model complexity.

``` text
Depth ↑
   ↓
Training Error ↓
   ↓
Generalization may ↓
```

------------------------------------------------------------------------

## ❌ Mistake 5: Forgetting cross-validation

A single train/test split can provide a noisy estimate.

Use:

``` python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print(scores)
print(scores.mean())
```

------------------------------------------------------------------------

# 22. 🧠 Best Practices

### 1. Start with a baseline

Train a simple tree first.

### 2. Control depth

Use:

``` python
max_depth
```

### 3. Control leaf size

Use:

``` python
min_samples_leaf
```

### 4. Use stratified splitting for classification

``` python
train_test_split(
    X,
    y,
    stratify=y,
    random_state=42
)
```

### 5. Use cross-validation

Compare models using the same validation strategy.

### 6. Tune hyperparameters

Use `GridSearchCV` or `RandomizedSearchCV`.

### 7. Inspect feature importance carefully

Feature importance is useful but should not automatically be interpreted
as causality.

### 8. Prefer simpler models when performance is similar

A smaller tree is generally easier to explain and maintain.

------------------------------------------------------------------------

# 23. 🚀 Advanced Concepts

## 🌲 23.1 Recursive Binary Splitting

Many Decision Tree implementations repeatedly perform binary splits.

``` text
Dataset
   |
   +---- Split 1
   |      |
   |      +---- Split 2
   |      |
   |      +---- Split 3
   |
   +---- Split 4
```

The procedure continues recursively.

------------------------------------------------------------------------

## 🧮 23.2 Greedy Algorithm

Decision Trees typically use a greedy strategy.

At each node:

1.  Evaluate candidate splits.
2.  Choose the best current split.
3.  Never revisit earlier decisions globally.

This makes training practical but does not guarantee the globally
optimal tree.

------------------------------------------------------------------------

## 🌳 23.3 Recursive Tree Growth

``` mermaid
flowchart TD
    A[Root Dataset] --> B[Find Best Feature]
    B --> C[Find Best Threshold]
    C --> D[Create Left Child]
    C --> E[Create Right Child]
    D --> F{Stop?}
    E --> G{Stop?}
    F -->|No| B
    G -->|No| B
    F -->|Yes| H[Leaf]
    G -->|Yes| I[Leaf]
```

------------------------------------------------------------------------

## 🔎 23.4 Threshold Selection

For a numerical feature:

``` text
Age:
20
25
30
35
40
```

Candidate thresholds can occur between sorted values, such as:

``` text
22.5
27.5
32.5
37.5
```

The algorithm evaluates candidate splits and selects one that provides
the best criterion improvement.

------------------------------------------------------------------------

## 🧠 23.5 Multiclass Classification

Decision Trees can naturally support more than two classes.

Example:

``` text
Target:
Cat
Dog
Bird
```

The splitting criterion evaluates all class proportions in the node.

------------------------------------------------------------------------

# 24. 🌲 Ensemble Methods Based on Trees

Single Decision Trees are the foundation for many powerful ensemble
algorithms.

``` mermaid
flowchart LR
    A[Decision Tree] --> B[Bagging]
    A --> C[Random Forest]
    A --> D[Gradient Boosting]
    D --> E[XGBoost]
    D --> F[LightGBM]
    D --> G[CatBoost]
```

## 🌲 Random Forest

Builds many randomized Decision Trees and combines their predictions.

``` text
Tree 1 ──┐
Tree 2 ──┤
Tree 3 ──┤
Tree 4 ──┤──→ Voting / Averaging → Final Prediction
Tree 5 ──┘
```

## ⚡ Gradient Boosting

Builds trees sequentially, where later trees focus on correcting errors
made by earlier trees.

### Comparison

  Algorithm           Main Idea
  ------------------- --------------------------------------------------------
  Decision Tree       One tree
  Random Forest       Many independent randomized trees
  Gradient Boosting   Sequential error-correcting trees
  XGBoost             Optimized gradient boosting
  LightGBM            Efficient gradient boosting
  CatBoost            Gradient boosting with strong categorical-data support

------------------------------------------------------------------------

# 25. 🔍 Feature Importance

Decision Trees can estimate feature importance.

``` python
import pandas as pd

importance = pd.Series(
    model.feature_importances_,
    index=data.feature_names
).sort_values(ascending=False)

print(importance)
```

## 📊 Visualization

``` python
import matplotlib.pyplot as plt

importance.sort_values().plot(kind="barh")

plt.xlabel("Feature Importance")
plt.title("Decision Tree Feature Importance")
plt.show()
```

### Important warning

Feature importance tells you how useful a feature was for the trained
model's splits.

It does **not** prove:

-   Causality
-   Real-world importance
-   Business causation

For more robust interpretation, consider permutation importance or
SHAP-based analysis.

------------------------------------------------------------------------

# 26. 🧩 Handling Missing and Categorical Data

## Missing Values

Handling depends on the implementation.

A common workflow is:

``` text
Raw Data
   ↓
Detect Missing Values
   ↓
Impute / Handle Missing Values
   ↓
Train Tree
```

Example:

``` python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
```

Always fit preprocessing steps on training data only.

------------------------------------------------------------------------

## Categorical Variables

Depending on the implementation, categorical features may need encoding.

Common approaches:

-   One-Hot Encoding
-   Ordinal Encoding
-   Native categorical handling in suitable tree algorithms

Example:

``` python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    handle_unknown="ignore"
)

X_encoded = encoder.fit_transform(X)
```

------------------------------------------------------------------------

# 27. 🛠️ Practical Mini-Project

## 🎯 Project: Customer Purchase Prediction

### Objective

Predict whether a customer will purchase a product based on:

-   Age
-   Income
-   Website visits
-   Previous purchases
-   Time spent on website

### Project Workflow

``` mermaid
flowchart TD
    A[Customer Dataset] --> B[Data Cleaning]
    B --> C[EDA]
    C --> D[Feature Selection]
    D --> E[Train Test Split]
    E --> F[Decision Tree Training]
    F --> G[Hyperparameter Tuning]
    G --> H[Model Evaluation]
    H --> I[Feature Importance]
    I --> J[Final Prediction]
```

------------------------------------------------------------------------

## 📁 Example Dataset

``` text
age,income,visits,previous_purchase,purchase
22,30000,2,0,0
25,45000,5,1,1
35,70000,3,2,1
42,50000,1,0,0
29,60000,7,3,1
```

------------------------------------------------------------------------

## 🧪 Implementation

``` python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("customer_data.csv")

X = df.drop("purchase", axis=1)
y = df["purchase"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    min_samples_leaf=3,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

------------------------------------------------------------------------

## 🌳 Visualize the Tree

``` python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True
)

plt.show()
```

------------------------------------------------------------------------

## 🔬 Suggested Experiments

Try changing:

``` python
max_depth=2
```

Then:

``` python
max_depth=4
```

Then:

``` python
max_depth=8
```

Compare:

    Depth   Training Accuracy   Testing Accuracy Observation
  ------- ------------------- ------------------ -------------------
        2             Measure            Measure Possibly underfit
        4             Measure            Measure Often balanced
        8             Measure            Measure May overfit

The exact results depend on the dataset.

------------------------------------------------------------------------

# 28. 🎤 Interview Questions and Points

## Q1. What is a Decision Tree?

A Decision Tree is a supervised learning algorithm that makes
predictions using a hierarchy of feature-based decision rules.

------------------------------------------------------------------------

## Q2. What is the root node?

The root node is the first decision node of the tree.

------------------------------------------------------------------------

## Q3. What is a leaf node?

A leaf node is the terminal node containing the final prediction.

------------------------------------------------------------------------

## Q4. What is Gini impurity?

Gini impurity measures how mixed the classes are within a node.

$$
Gini = 1-\sum p_i^2
$$

------------------------------------------------------------------------

## Q5. What is entropy?

Entropy measures uncertainty or disorder in a node.

$$
Entropy=-\sum p_i\log_2(p_i)
$$

------------------------------------------------------------------------

## Q6. What is Information Gain?

Information Gain measures how much entropy decreases after a split.

------------------------------------------------------------------------

## Q7. Why do Decision Trees overfit?

Because unrestricted trees can keep splitting until they memorize very
specific training examples.

------------------------------------------------------------------------

## Q8. How do you reduce overfitting?

Use:

-   `max_depth`
-   `min_samples_split`
-   `min_samples_leaf`
-   `max_leaf_nodes`
-   `ccp_alpha`
-   Cross-validation
-   Pruning

------------------------------------------------------------------------

## Q9. Does a Decision Tree require feature scaling?

Generally, no.

Decision Trees are based on feature thresholds and are usually
insensitive to monotonic scaling.

------------------------------------------------------------------------

## Q10. What is pruning?

Pruning removes unnecessary branches to reduce model complexity and
improve generalization.

------------------------------------------------------------------------

## Q11. What is the difference between Random Forest and Decision Tree?

  Decision Tree                      Random Forest
  ---------------------------------- --------------------------------
  One tree                           Many trees
  High variance                      Lower variance
  Easier to interpret                Less interpretable
  Can overfit easily                 Generally more robust
  Faster/simple for small problems   More computationally expensive

------------------------------------------------------------------------

## Q12. Why are Decision Trees called non-parametric models?

Because they do not assume a fixed parametric functional form such as:

$$
y = \beta_0 + \beta_1x
$$

Instead, they learn decision regions directly from the data.

------------------------------------------------------------------------

# 29. 📝 Quick Revision

## ⚡ Key Concepts

``` text
Decision Tree
│
├── Supervised Learning
│
├── Classification
│   ├── Gini
│   ├── Entropy
│   └── Information Gain
│
├── Regression
│   ├── MSE
│   └── MAE
│
├── Nodes
│   ├── Root
│   ├── Internal
│   └── Leaf
│
├── Splitting
│   └── Feature + Threshold
│
├── Overfitting
│   └── Pruning / Hyperparameters
│
└── Ensembles
    ├── Random Forest
    ├── Gradient Boosting
    ├── XGBoost
    ├── LightGBM
    └── CatBoost
```

------------------------------------------------------------------------

## 📐 Important Formulas

### Gini Impurity

$$
Gini = 1-\sum p_i^2
$$

### Entropy

$$
Entropy=-\sum p_i\log_2(p_i)
$$

### Information Gain

$$
IG=Entropy(parent)-WeightedEntropy(children)
$$

### Mean Squared Error

$$
MSE=\frac{1}{n}\sum(y_i-\hat y_i)^2
$$

### Mean Absolute Error

$$
MAE=\frac{1}{n}\sum|y_i-\hat y_i|
$$

### Accuracy

$$
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
$$

### Precision

$$
Precision=\frac{TP}{TP+FP}
$$

### Recall

$$
Recall=\frac{TP}{TP+FN}
$$

### F1 Score

$$
F1=2\frac{Precision\times Recall}{Precision+Recall}
$$

------------------------------------------------------------------------

## 💻 Important Commands

### Classification

``` python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

### Regression

``` python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(
    criterion="squared_error",
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

### Plot Tree

``` python
from sklearn.tree import plot_tree

plot_tree(
    model,
    feature_names=X.columns,
    filled=True
)
```

### Feature Importance

``` python
model.feature_importances_
```

### Cost-Complexity Path

``` python
path = model.cost_complexity_pruning_path(
    X_train,
    y_train
)

ccp_alphas = path.ccp_alphas
```

------------------------------------------------------------------------

# 30. 🗺️ Visual Roadmap

``` mermaid
flowchart TD
    A[Machine Learning] --> B[Supervised Learning]
    B --> C[Decision Tree]

    C --> D[Classification]
    C --> E[Regression]

    D --> F[Gini]
    D --> G[Entropy]
    D --> H[Information Gain]

    E --> I[MSE]
    E --> J[MAE]

    C --> K[Splitting]
    C --> L[Stopping]
    C --> M[Pruning]

    K --> N[Feature]
    K --> O[Threshold]

    M --> P[Pre-Pruning]
    M --> Q[Post-Pruning]

    C --> R[Evaluation]
    R --> S[Accuracy]
    R --> T[Precision]
    R --> U[Recall]
    R --> V[F1]

    C --> W[Ensemble Learning]
    W --> X[Random Forest]
    W --> Y[Gradient Boosting]
    Y --> Z[XGBoost / LightGBM / CatBoost]
```

------------------------------------------------------------------------

# 🎯 Final Takeaways

1.  **Decision Trees are supervised learning algorithms.**
2.  They can solve both **classification and regression** problems.
3.  They make predictions using a sequence of **decision rules**.
4.  The first node is called the **root**.
5.  Final predictions are made at **leaf nodes**.
6.  Classification trees commonly use **Gini impurity or Entropy**.
7.  Information Gain measures the reduction in entropy.
8.  Regression trees commonly minimize **MSE or related loss criteria**.
9.  Decision Trees generally **do not require feature scaling**.
10. Deep trees can easily **overfit**.
11. `max_depth`, `min_samples_leaf`, and `ccp_alpha` are important
    complexity controls.
12. Pruning can improve generalization.
13. Feature importance can help understand the trained model, but it is
    not proof of causality.
14. Decision Trees are easy to visualize and explain.
15. Single trees can have high variance.
16. **Random Forest** reduces variance by combining many trees.
17. **Gradient Boosting** builds trees sequentially to correct errors.
18. Cross-validation is useful for reliable model comparison.
19. Always evaluate performance on unseen data.
20. A good Decision Tree balances **simplicity, predictive performance,
    and generalization**.

------------------------------------------------------------------------

## 🚀 Learning Path

``` text
Decision Tree
     ↓
Understand Nodes & Splits
     ↓
Learn Gini & Entropy
     ↓
Understand Information Gain
     ↓
Classification Tree
     ↓
Regression Tree
     ↓
Overfitting
     ↓
Pruning
     ↓
Hyperparameter Tuning
     ↓
Feature Importance
     ↓
Random Forest
     ↓
Gradient Boosting
     ↓
XGBoost / LightGBM / CatBoost
     ↓
Advanced Tree-Based ML
```

> 🌳 **Remember:** A Decision Tree learns a set of simple decisions that
> together create a powerful predictive model. The key to a good tree is
> not making it as deep as possible---it is finding the right balance
> between **fit and generalization**.
