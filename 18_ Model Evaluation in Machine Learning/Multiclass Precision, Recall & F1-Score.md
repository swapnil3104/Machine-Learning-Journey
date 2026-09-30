# 📊 Multiclass Precision, Recall & F1-Score

## 📌 Table of Contents

1. [Introduction](#-introduction)
2. [Why Accuracy Is Not Always Enough](#-why-accuracy-is-not-always-enough)
3. [Confusion Matrix](#-confusion-matrix)
4. [TP, TN, FP and FN in Multiclass Classification](#-tp-tn-fp-and-fn-in-multiclass-classification)
5. [Precision](#-precision)
6. [Recall](#-recall)
7. [F1-Score](#-f1-score)
8. [Example of Multiclass Classification](#-example-of-multiclass-classification)
9. [Class-Wise Calculation](#-class-wise-calculation)
10. [Macro Average](#-macro-average)
11. [Micro Average](#-micro-average)
12. [Weighted Average](#-weighted-average)
13. [Comparison of Averaging Methods](#-comparison-of-averaging-methods)
14. [Python Implementation](#-python-implementation)
15. [Using Classification Report](#-using-classification-report)
16. [Important Practical Points](#-important-practical-points)
17. [Advantages and Limitations](#-advantages-and-limitations)
18. [Summary](#-summary)

---

# 🚀 Introduction

When building a classification model, **accuracy alone does not tell the complete story**.

For example, suppose a dataset contains:

* 90% Class A
* 5% Class B
* 5% Class C

A model that predicts every sample as Class A can achieve approximately **90% accuracy**, while completely failing to identify Classes B and C.

Therefore, classification models are commonly evaluated using:

* 🎯 Precision
* 🔍 Recall
* ⚖️ F1-Score
* 📊 Confusion Matrix
* 📈 Accuracy

For multiclass classification, Precision, Recall and F1-Score can be calculated **for every class independently** and then combined using different averaging strategies.

---

# 🎯 Why Accuracy Is Not Always Enough

Consider a three-class classification problem:

```text
Classes:
Cat
Dog
Horse
```

Suppose there are 1000 images:

```text
Cat    = 800
Dog    = 100
Horse  = 100
```

If the model predicts every image as Cat:

```text
Accuracy = 800 / 1000
         = 80%
```

80% may look good.

However:

```text
Dog detection  → Very poor
Horse detection → Very poor
```

This is why we need **Precision, Recall and F1-Score**.

---

# 📊 Confusion Matrix

A confusion matrix compares:

```text
Actual Class
     ↓
Predicted Class
```

For a 3-class problem:

| Actual \ Predicted | Cat | Dog | Horse |
| ------------------ | --: | --: | ----: |
| Cat                |  40 |   5 |     5 |
| Dog                |   3 |  42 |     5 |
| Horse              |   2 |   3 |    45 |

The diagonal represents correct predictions:

```text
Cat → Cat     = 40
Dog → Dog     = 42
Horse → Horse = 45
```

Off-diagonal values represent incorrect predictions.

---

# 🔢 TP, TN, FP and FN in Multiclass Classification

In multiclass classification, Precision and Recall are generally calculated using a **One-vs-Rest (OvR)** approach.

For each class:

```text
Current class = Positive
All other classes = Negative
```

For example, when evaluating **Cat**:

```text
Cat       → Positive
Dog       → Negative
Horse     → Negative
```

### True Positive — TP

Samples that actually belong to the class and were predicted as that class.

### False Positive — FP

Samples that belong to another class but were incorrectly predicted as the current class.

### False Negative — FN

Samples that belong to the current class but were predicted as another class.

### True Negative — TN

Samples that belong to other classes and were correctly predicted as other classes.

---

# 🎯 Precision

Precision answers:

> **"Of all samples predicted as this class, how many were actually this class?"**

Formula:

```text
Precision = TP / (TP + FP)
```

According to scikit-learn, precision is the ratio of true positives to true positives plus false positives.

### Example

Suppose:

```text
TP = 40
FP = 10
```

Then:

```text
Precision = 40 / (40 + 10)
          = 40 / 50
          = 0.80
```

Therefore:

```text
Precision = 80%
```

### Interpretation

High precision means:

```text
When the model predicts this class,
it is usually correct.
```

---

# 🔍 Recall

Recall answers:

> **"Of all actual samples belonging to this class, how many did the model correctly identify?"**

Formula:

```text
Recall = TP / (TP + FN)
```

Scikit-learn defines recall as the ability of the classifier to find the positive samples.

### Example

Suppose:

```text
TP = 40
FN = 5
```

Then:

```text
Recall = 40 / (40 + 5)
       = 40 / 45
       = 0.8889
```

Therefore:

```text
Recall ≈ 88.89%
```

### Interpretation

High recall means:

```text
The model is successfully finding
most of the actual samples of this class.
```

---

# ⚖️ F1-Score

F1-Score combines Precision and Recall.

Formula:

```text
F1 = 2 × (Precision × Recall)
         ---------------------
         (Precision + Recall)
```

Or:

```text
F1 = 2PR / (P + R)
```

### Example

Suppose:

```text
Precision = 0.80
Recall    = 0.8889
```

Then:

```text
F1 = 2 × (0.80 × 0.8889)
     --------------------
        0.80 + 0.8889

F1 ≈ 0.8421
```

Therefore:

```text
F1 ≈ 84.21%
```

F1 is the harmonic mean of precision and recall.

---

# 🧠 Example of Multiclass Classification

Suppose we have:

```text
Class 0 → Cat
Class 1 → Dog
Class 2 → Horse
```

Actual and predicted values:

```python
y_true = [
    "Cat",
    "Cat",
    "Cat",
    "Dog",
    "Dog",
    "Dog",
    "Horse",
    "Horse",
    "Horse"
]

y_pred = [
    "Cat",
    "Cat",
    "Dog",
    "Dog",
    "Dog",
    "Horse",
    "Horse",
    "Horse",
    "Cat"
]
```

Confusion matrix:

| Actual \ Predicted | Cat | Dog | Horse |
| ------------------ | --: | --: | ----: |
| Cat                |   2 |   1 |     0 |
| Dog                |   0 |   2 |     1 |
| Horse              |   1 |   0 |     2 |

---

# 🧮 Class-Wise Calculation

## 🐱 Class: Cat

From the confusion matrix:

```text
TP = 2
FP = 1 + 0 = 1
FN = 1 + 0 = 1
```

### Precision

```text
Precision = 2 / (2 + 1)
          = 0.6667
```

### Recall

```text
Recall = 2 / (2 + 1)
       = 0.6667
```

### F1

```text
F1 = 2 × (0.6667 × 0.6667)
     -----------------------
       0.6667 + 0.6667

F1 = 0.6667
```

---

## 🐶 Class: Dog

```text
TP = 2
FP = 1
FN = 1
```

Therefore:

```text
Precision = 2 / 3
          = 0.6667

Recall = 2 / 3
       = 0.6667

F1 = 0.6667
```

---

## 🐴 Class: Horse

```text
TP = 2
FP = 1
FN = 1
```

Therefore:

```text
Precision = 0.6667
Recall    = 0.6667
F1        = 0.6667
```

---

# 📊 Class-Wise Results

| Class | Precision | Recall | F1-Score |
| ----- | --------: | -----: | -------: |
| Cat   |     0.667 |  0.667 |    0.667 |
| Dog   |     0.667 |  0.667 |    0.667 |
| Horse |     0.667 |  0.667 |    0.667 |

These are **per-class metrics**.

Now we can calculate overall multiclass metrics.

---

# 📊 Macro Average

Macro averaging calculates the metric separately for every class and then takes the **unweighted mean**.

Scikit-learn describes macro averaging as calculating the metric for each label and taking their unweighted mean.

### Macro Precision

```text
Macro Precision =
(Pcat + Pdog + Phorse) / 3
```

Using our example:

```text
= (0.667 + 0.667 + 0.667) / 3

= 0.667
```

### Macro Recall

```text
Macro Recall =
(Rcat + Rdog + Rhorse) / 3

= (0.667 + 0.667 + 0.667) / 3

= 0.667
```

### Macro F1

```text
Macro F1 =
(F1cat + F1dog + F1horse) / 3

= 0.667
```

### Important

Macro averaging gives:

```text
Equal importance to every class
```

It is particularly useful when you want to understand performance across all classes without letting large classes dominate the result.

---

# 🌍 Micro Average

Micro averaging combines all class-level TP, FP and FN values **before** calculating the metric.

Formula:

```text
Micro Precision =
ΣTP / (ΣTP + ΣFP)
```

```text
Micro Recall =
ΣTP / (ΣTP + ΣFN)
```

For single-label multiclass classification, micro-averaged precision, recall and F1 are equal to overall accuracy when all classes are included.

### Example

Suppose:

```text
Total TP = 6
Total FP = 3
Total FN = 3
```

Then:

```text
Micro Precision = 6 / (6 + 3)
                = 0.667

Micro Recall = 6 / (6 + 3)
             = 0.667

Micro F1 = 0.667
```

---

# ⚖️ Weighted Average

Weighted averaging calculates the metric for each class and then weights each class according to its **support** — the number of actual samples belonging to that class.

Formula:

```text
Weighted Metric =
Σ(Metric × Class Support)
-------------------------
       Total Support
```

### Example

Suppose:

| Class |   F1 | Support |
| ----- | ---: | ------: |
| Cat   | 0.90 |     100 |
| Dog   | 0.70 |      30 |
| Horse | 0.60 |      20 |

Weighted F1:

```text
= (0.90 × 100 + 0.70 × 30 + 0.60 × 20)
  --------------------------------------
                 150

= (90 + 21 + 12) / 150

= 123 / 150

= 0.82
```

Therefore:

```text
Weighted F1 = 82%
```

The larger class contributes more to the final score.

---

# 📋 Comparison of Averaging Methods

| Method   | How it works                  | Class Imbalance               | Main Use                       |
| -------- | ----------------------------- | ----------------------------- | ------------------------------ |
| Macro    | Average class metrics equally | Sensitive                     | Treat every class equally      |
| Micro    | Combine TP/FP/FN globally     | Majority classes can dominate | Overall performance            |
| Weighted | Weight by class support       | Accounts for imbalance        | Real-world imbalanced datasets |
| None     | Return each class separately  | Shows individual performance  | Detailed analysis              |

---

# 💻 Python Implementation

Install scikit-learn:

```bash
pip install scikit-learn
```

Import the required functions:

```python
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)
```

Define actual and predicted values:

```python
y_true = [
    "Cat", "Cat", "Cat",
    "Dog", "Dog", "Dog",
    "Horse", "Horse", "Horse"
]

y_pred = [
    "Cat", "Cat", "Dog",
    "Dog", "Dog", "Horse",
    "Horse", "Horse", "Cat"
]
```

---

## 🔹 Per-Class Metrics

```python
precision = precision_score(
    y_true,
    y_pred,
    average=None
)

recall = recall_score(
    y_true,
    y_pred,
    average=None
)

f1 = f1_score(
    y_true,
    y_pred,
    average=None
)

print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
```

`average=None` returns the metric for each class separately.

---

# 🔹 Macro Average

```python
precision_macro = precision_score(
    y_true,
    y_pred,
    average="macro"
)

recall_macro = recall_score(
    y_true,
    y_pred,
    average="macro"
)

f1_macro = f1_score(
    y_true,
    y_pred,
    average="macro"
)

print("Macro Precision:", precision_macro)
print("Macro Recall:", recall_macro)
print("Macro F1:", f1_macro)
```

---

# 🔹 Micro Average

```python
precision_micro = precision_score(
    y_true,
    y_pred,
    average="micro"
)

recall_micro = recall_score(
    y_true,
    y_pred,
    average="micro"
)

f1_micro = f1_score(
    y_true,
    y_pred,
    average="micro"
)

print("Micro Precision:", precision_micro)
print("Micro Recall:", recall_micro)
print("Micro F1:", f1_micro)
```

---

# 🔹 Weighted Average

```python
precision_weighted = precision_score(
    y_true,
    y_pred,
    average="weighted"
)

recall_weighted = recall_score(
    y_true,
    y_pred,
    average="weighted"
)

f1_weighted = f1_score(
    y_true,
    y_pred,
    average="weighted"
)

print("Weighted Precision:", precision_weighted)
print("Weighted Recall:", recall_weighted)
print("Weighted F1:", f1_weighted)
```

---

# 📑 Using Classification Report

Instead of calculating everything separately, you can use:

```python
from sklearn.metrics import classification_report

print(
    classification_report(
        y_true,
        y_pred
    )
)
```

Example output:

```text
              precision    recall  f1-score   support

         Cat       0.67      0.67      0.67         3
         Dog       0.67      0.67      0.67         3
       Horse       0.67      0.67      0.67         3

    accuracy                           0.67         9
   macro avg       0.67      0.67      0.67         9
weighted avg       0.67      0.67      0.67         9
```

---

# 🧠 Understanding `support`

Support means:

> **Number of actual samples belonging to a particular class.**

For example:

```text
Cat    → 100 samples
Dog    → 50 samples
Horse  → 20 samples
```

Then:

```text
Cat support    = 100
Dog support    = 50
Horse support  = 20
```

Support is particularly important when interpreting **weighted averages**.

---

# 📌 Precision vs Recall

| Metric    | Question                                                   |
| --------- | ---------------------------------------------------------- |
| Precision | When the model predicts positive, how often is it correct? |
| Recall    | Of all actual positives, how many did the model find?      |
| F1        | How well are precision and recall balanced?                |

### Easy Memory Trick

```text
Precision → Prediction correctness
Recall    → Finding actual positives
F1        → Balance between both
```

---

# 🏥 Example: Medical Classification

Suppose an AI model detects:

```text
Normal
Cataract
Glaucoma
Diabetic Retinopathy
```

For **Glaucoma**:

```text
TP = 80
FP = 10
FN = 20
```

### Precision

```text
Precision = 80 / (80 + 10)

           = 0.8889

           = 88.89%
```

### Recall

```text
Recall = 80 / (80 + 20)

       = 0.80

       = 80%
```

### F1

```text
F1 = 2 × (0.8889 × 0.80)
     ---------------------
       0.8889 + 0.80

   ≈ 0.8421

   ≈ 84.21%
```

This tells us that the model's Glaucoma performance should be considered separately from its performance on other diseases.

---

# ⚠️ Important: Accuracy vs Macro F1

Suppose:

```text
Accuracy = 95%
Macro F1 = 72%
```

This can happen when:

```text
Majority classes → Very good performance
Minority classes → Poor performance
```

Therefore, don't automatically conclude that the model performs well based only on accuracy.

Always inspect:

```text
✓ Confusion Matrix
✓ Per-class Precision
✓ Per-class Recall
✓ Per-class F1
✓ Macro F1
✓ Weighted F1
```

---

# 🔥 When Should You Use Which Metric?

## Precision

Use precision when:

```text
False Positives are costly.
```

Example:

```text
Spam detection
```

You don't want legitimate emails incorrectly classified as spam.

---

## Recall

Use recall when:

```text
False Negatives are costly.
```

Example:

```text
Disease detection
```

You want to identify as many actual disease cases as possible.

---

## F1-Score

Use F1 when:

```text
You need a balance between Precision and Recall.
```

---

## Macro F1

Useful when:

```text
Every class is important.
```

Especially when classes have different numbers of samples.

---

## Weighted F1

Useful when:

```text
Class distribution is imbalanced
and you want class support reflected in the overall score.
```

---

# 🧮 Complete Calculation Workflow

For a multiclass classification problem:

```text
              Dataset
                 ↓
        Actual + Predictions
                 ↓
          Confusion Matrix
                 ↓
       ┌─────────┴─────────┐
       ↓                   ↓
    Class 1              Class 2 ... Class K
       ↓                   ↓
 TP / FP / FN          TP / FP / FN
       ↓                   ↓
 Precision             Precision
 Recall                Recall
 F1                    F1
       └─────────┬─────────┘
                 ↓
        Averaging Strategy
                 ↓
      ┌──────────┼──────────┐
      ↓          ↓          ↓
    Macro      Micro     Weighted
```

---

# 📝 Important Formulas Cheat Sheet

### Precision

```text
Precision = TP / (TP + FP)
```

### Recall

```text
Recall = TP / (TP + FN)
```

### F1-Score

```text
F1 = 2 × Precision × Recall
     ------------------------
      Precision + Recall
```

### Macro

```text
Macro =
Sum of class metrics / Number of classes
```

### Weighted

```text
Weighted =
Σ(metric × support) / total support
```

### Micro

```text
Micro Precision =
ΣTP / (ΣTP + ΣFP)

Micro Recall =
ΣTP / (ΣTP + ΣFN)
```

---

# ⚠️ Common Mistakes

### ❌ Mistake 1: Using Binary Average

For multiclass classification, don't blindly use:

```python
average="binary"
```

Instead use:

```python
average=None
average="macro"
average="micro"
average="weighted"
```

depending on the analysis required.

---

### ❌ Mistake 2: Looking Only at Accuracy

Always inspect class-wise metrics.

---

### ❌ Mistake 3: Ignoring Class Imbalance

If one class has 10,000 samples and another has 100 samples, accuracy may hide poor minority-class performance.

---

### ❌ Mistake 4: Confusing Precision and Recall

Remember:

```text
Precision → Predicted Positive → How many correct?

Recall → Actual Positive → How many found?
```

---

### ❌ Mistake 5: Reporting Only One F1 Score

For a multiclass model, report at least:

```text
Per-class F1
Macro F1
Weighted F1
```

when class imbalance or class-specific performance matters.

---

# 📊 Recommended Evaluation Table

For a machine learning project, a useful report is:

| Class            | Precision |   Recall | F1-Score |  Support |
| ---------------- | --------: | -------: | -------: | -------: |
| Class A          |      0.91 |     0.88 |     0.89 |      500 |
| Class B          |      0.84 |     0.87 |     0.85 |      300 |
| Class C          |      0.76 |     0.81 |     0.78 |      200 |
| **Macro Avg**    |  **0.84** | **0.85** | **0.84** | **1000** |
| **Weighted Avg** |  **0.87** | **0.86** | **0.86** | **1000** |

This makes the model's performance much easier to interpret.

---

# 🚀 Best Practices

For multiclass classification:

```text
1. Calculate the confusion matrix
2. Calculate per-class Precision
3. Calculate per-class Recall
4. Calculate per-class F1
5. Check class support
6. Calculate Macro averages
7. Calculate Weighted averages
8. Check Micro metrics when appropriate
9. Investigate poorly performing classes
10. Don't rely only on accuracy
```

---

# 📚 Key Takeaways

```text
Precision
    ↓
How correct are my positive predictions?

Recall
    ↓
How many actual positives did I find?

F1-Score
    ↓
How balanced are Precision and Recall?

Macro Average
    ↓
Every class gets equal importance.

Micro Average
    ↓
All predictions are combined globally.

Weighted Average
    ↓
Classes are weighted according to their support.
```

### ⭐ One-Line Memory Trick

> **Precision = Correctness of predictions, Recall = Coverage of actual positives, F1 = Balance between Precision and Recall.**

For multiclass classification, Precision, Recall and F1 can be calculated independently for each class and then combined using `macro`, `micro`, or `weighted` averaging.

---

## 🔗 References

* Scikit-learn — `precision_recall_fscore_support`: definitions, formulas, averaging strategies, and support.
* Scikit-learn — Multiclass and multilabel model evaluation.
* Microsoft Azure — Multiclass metrics and confusion matrix interpretation.
