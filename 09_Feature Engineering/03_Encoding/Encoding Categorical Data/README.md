# 🔤 Encoding Categorical Data in Machine Learning

## 📌 Module Overview
Welcome to the **Categorical Data Encoding** sub-module! Machine learning models require numerical feature inputs. This folder covers techniques for converting nominal and ordinal categories into structured numerical vectors.

### Why is this topic important?
Improper encoding can either introduce false ordinal hierarchies (e.g., arbitrarily encoding cities as 1, 2, 3) or cause high-dimensional sparsity and dummy variable traps (e.g., unbounded One-Hot Encoding on high cardinality features).

### What You Will Learn
- Nominal encoding: One-Hot Encoding (OHE) with `drop='first'` to avoid collinearity.
- Ordinal encoding: Preserving natural ordering (e.g., Education, Ratings, T-shirt sizes).
- Handling infrequent / rare categories and unseen labels during inference.

---

## 📂 File Structure

```text
Encoding Categorical Data/
├── One Hot Encoding/
│   ├── cars.csv
│   └── one-hot-encoding.ipynb
├── Ordinal Encoding/
│   ├── customer.csv
│   └── Ordinal Encoding.ipynb
└── README.md
```

---

> **Categorical Encoding** is the process of converting categorical (text) data into numerical values so that machine learning algorithms can understand and process it.

---

# 📌 Why Do We Need Encoding?

Most machine learning algorithms work only with **numerical data**.

For example, a model cannot understand values like:

| Gender |
|---------|
| Male |
| Female |
| Female |

Instead, we convert them into numbers using different encoding techniques.

---

# 📊 Types of Categorical Encoding

```text
                     Encoding Categorical Data
                               │
      ┌────────────────────────┼────────────────────────┐
      │                        │                        │
      ▼                        ▼                        ▼
 One-Hot Encoding       Label Encoding         Target Encoding
                              │
                              ▼
                     Ordinal Encoding
                     (Special case of Label Encoding)
```

---

# 🌳 Visual Hierarchy

```text
                  Categorical Data
                         │
                         ▼
                Encoding Techniques
                         │
     ┌──────────┬─────────────┬─────────────┬──────────────┐
     │          │             │             │
     ▼          ▼             ▼             ▼
 One-Hot     Label        Ordinal      Target
 Encoding    Encoding     Encoding     Encoding
```

---

# 1️⃣ One-Hot Encoding

## Definition

One-Hot Encoding creates a **new binary column for every category**.

Each category is represented using **0** or **1**.

### Example Dataset

| Name | Gender |
|------|---------|
| John | Male |
| Emma | Female |
| Rachel | Female |

↓

### Encoded Dataset

| Name | Female | Male |
|------|---------|------|
| John | 0 | 1 |
| Emma | 1 | 0 |
| Rachel | 1 | 0 |

---

## Visual Representation

```text
Male
 ↓
[0 1]

Female
 ↓
[1 0]
```

---

## Advantages

✔ No ordinal relationship

✔ Works well for nominal data

✔ Easy to understand

---

## Disadvantages

❌ Creates many new columns

❌ High-dimensional data

---

## Best Used For

- Gender
- Color
- Country
- City (small number of categories)

---

# 2️⃣ Label Encoding

## Definition

Each category is assigned a unique integer.

### Example

| Location |
|----------|
| New York |
| California |
| Texas |

↓

| Location | Encoded |
|----------|----------|
| New York | 0 |
| California | 1 |
| Texas | 2 |

---

## Visual Representation

```text
New York
      │
      ▼
      0

California
      │
      ▼
      1

Texas
      │
      ▼
      2
```

---

## Problem

The model may assume

```text
0 < 1 < 2

New York < California < Texas
```

which is **not true** because these categories have **no natural order**.

---

## Advantages

✔ Very simple

✔ Uses only one column

✔ Memory efficient

---

## Disadvantages

❌ Introduces false ordering

---

## Best Used For

- Tree-based models
- Binary categories
- When ordering is acceptable

---

# 3️⃣ Ordinal Encoding

## Definition

Ordinal Encoding is used when categories have a **natural order**.

### Example

| Education |
|------------|
| UG |
| PG |
| PhD |

↓

| Education | Encoded |
|------------|----------|
| UG | 1 |
| PG | 2 |
| PhD | 3 |

---

## Visual Representation

```text
UG
 │
 ▼
 1
 │
 ▼
PG
 │
 ▼
 2
 │
 ▼
PhD
 │
 ▼
 3
```

---

## Why It Works

Education has an actual ranking.

```text
UG
   ↓

PG
   ↓

PhD
```

Unlike cities or colors.

---

## Advantages

✔ Preserves order

✔ Only one column

✔ Useful for ranking problems

---

## Disadvantages

❌ Only works when order exists

---

## Best Used For

- Education Level
- Satisfaction Rating
- T-Shirt Size
- Experience Level

---

# 4️⃣ Target Encoding

## Definition

Replace each category with the **mean of the target variable**.

---

### Original Dataset

| City | House Price |
|------|-------------|
| NY | 200 |
| CA | 300 |
| TX | 500 |
| CA | 100 |
| TX | 400 |

---

### Calculate Mean

```text
NY

200

↓

Mean = 200

----------------

CA

300

100

↓

Mean = 200

----------------

TX

500

400

↓

Mean = 450
```

---

### Encoded Dataset

| City | Encoded |
|------|----------|
| NY | 200 |
| CA | 200 |
| TX | 450 |

---

## Visual Representation

```text
Category

↓

Average Target Value

↓

Encoded Number
```

---

## Advantages

✔ Excellent for high-cardinality data

✔ Uses only one column

✔ Often improves prediction

---

## Disadvantages

❌ Can cause data leakage

❌ Needs cross-validation

---

## Best Used For

- ZIP Codes
- Product IDs
- Customer IDs
- Thousands of categories

---

# 📊 Comparison of Encoding Techniques

| Technique | Creates New Columns | Preserves Order | Suitable For |
|------------|--------------------|-----------------|--------------|
| One-Hot Encoding | ✅ Yes | ❌ No | Nominal Data |
| Label Encoding | ❌ No | ⚠️ Artificial Order | Binary / Tree Models |
| Ordinal Encoding | ❌ No | ✅ Yes | Ordered Categories |
| Target Encoding | ❌ No | Based on Target | High Cardinality |

---

# 🎯 Choosing the Right Encoding

```text
                     Categorical Feature
                             │
               ┌─────────────┴─────────────┐
               │                           │
        Has Natural Order?           No Natural Order
               │                           │
         ┌─────┴─────┐                     │
         │           │                     │
        Yes          No                    │
         │           │                     │
         ▼           ▼                     ▼
 Ordinal Encoding  Few Categories?   Many Categories?
                     │                     │
               ┌─────┴─────┐         ┌─────┴─────┐
               │           │         │           │
              Yes          No        Yes         No
               │           │         │           │
               ▼           ▼         ▼           ▼
       One-Hot Encoding  Label   Target     One-Hot
                         Encoding Encoding  Encoding
```

---

# 💡 Summary

| Encoding Method | Best Use Case |
|-----------------|---------------|
| 🟢 One-Hot Encoding | Nominal categories with few unique values |
| 🔵 Label Encoding | Binary categories or tree-based algorithms |
| 🟠 Ordinal Encoding | Categories with meaningful order |
| 🔴 Target Encoding | High-cardinality categorical features |

---

# ✅ Key Takeaways

- Machine learning models require **numerical input**.
- **One-Hot Encoding** is the safest choice for most nominal categorical features.
- **Label Encoding** should not be used for unordered categories because it introduces false rankings.
- **Ordinal Encoding** is appropriate only when categories have a natural order.
- **Target Encoding** is powerful for datasets with many unique categories but must be applied carefully to avoid data leakage.
```
