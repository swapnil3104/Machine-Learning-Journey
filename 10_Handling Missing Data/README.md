# 🧩 10. Handling Missing Data & Imputation Techniques

## 📌 Folder Overview
Welcome to the **Handling Missing Data** module! Real-world datasets are almost never complete. This folder provides a comprehensive suite of strategies for analyzing, removing, and imputing missing data values across numerical and categorical features.

### Why is this topic important?
Most Scikit-Learn estimators cannot handle missing values (`NaN` / `None`) natively. Deleting rows blindly can cause severe information loss and introduce selection bias, while incorrect imputation can distort feature distributions and correlations. Understanding missing data mechanisms enables you to choose the optimal imputation strategy for your dataset.

### What You Will Learn
- Understanding Missing Data Mechanisms: **MCAR** (Missing Completely at Random), **MAR** (Missing at Random), and **MNAR** (Missing Not at Random).
- **Complete Case Analysis (CCA)**: When and how to safely drop missing rows/columns.
- **Univariate Imputation**: Mean/Median Imputation, Arbitrary Value Imputation, Mode/Frequent Category Imputation, and Missing Category Flagging.
- **Multivariate Imputation**: Advanced statistical techniques including K-Nearest Neighbors Imputer (**KNNImputer**) and Multivariate Imputation by Chained Equations (**IterativeImputer** / MICE).

---

## 📚 Topics Covered

Organized logically from basic dropping strategies to advanced multivariate imputation:

### 🟢 Beginner
- **Missing Data Diagnostics**: Identifying null counts, missing percentage thresholds, missingness heatmaps.
- **Complete Case Analysis (CCA)**: Dropping rows when missingness is $< 5\%$ and MCAR.

### 🟡 Intermediate
- **Simple Numerical Imputation**:
  - **Mean / Median Imputation**: Preserving central tendency when data is Gaussian vs skewed.
  - **Arbitrary Value Imputation**: Imputing outliers (e.g., `-999` or `999`) to signal missingness.
- **Simple Categorical Imputation**:
  - **Frequent Category (Mode) Imputation**: Replacing missing entries with the most frequent class.
  - **Missing Indicator / 'Missing' Category**: Explicitly creating a new category label.

### 🔴 Advanced
- **KNN Imputer**: Using distance-based spatial proximity ($k$-nearest neighbors) to estimate missing feature values based on similar samples.
- **Iterative Imputer (MICE)**: Modeling each feature with missing values as a function of all other features in a chained regression loop.

| Method / Strategy | Folder / Notebook | Target Data Type | Level |
| :--- | :--- | :--- | :--- |
| **Complete Case Analysis** | `01_Remove Valus/Example.ipynb` | Numerical / Categorical ($< 5\%$ missing) | Beginner |
| **Numerical Simple Imputer** | `02_Simple Imputer/Numerical Data/` | Continuous numerical features | Intermediate |
| **Categorical Simple Imputer** | `02_Simple Imputer/Categorical Data/` | Discrete categorical features | Intermediate |
| **KNN Imputer** | `03_KNN Imputer/` | Multidimensional numerical datasets | Advanced |
| **Iterative Imputer (MICE)** | `04_Iterative Imputer/` | Complex multivariate datasets | Advanced |

---

## 📂 File Structure

```text
10_Handling Missing Data/
├── README.md
├── Imputation_Notes.md
├── Notes.md
├── 01_Remove Valus/
│   ├── Example.ipynb
│   └── data_science_job.csv
├── 02_Simple Imputer/
│   ├── Categorical Data/
│   │   ├── frequent-value-imputation.ipynb
│   │   ├── missing-category-imputation.ipynb
│   │   └── train.csv
│   └── Numerical Data/
│       ├── arbitrary-value-imputation.ipynb
│       ├── mean-median-imputation.ipynb
│       └── titanic_toy.csv
├── 03_KNN Imputer/
└── 04_Iterative Imputer/
```
