# 🚨 12. Outlier Detection and Treatment

## 📌 Folder Overview
Welcome to the **Outliers in Machine Learning** module! Outliers are data points that deviate significantly from the rest of the observations in a dataset. This folder covers the statistical techniques used to detect, visualize, trim, and cap outliers across Gaussian and skewed feature distributions.

### Why is this topic important?
Outliers heavily distort statistical metrics (mean, variance) and degrade performance in algorithms sensitive to distance or squared errors (such as Linear Regression, Logistic Regression, KNN, and Neural Networks). Proper outlier treatment stabilizes model parameters and improves generalization on unseen data.

### What You Will Learn
- Visualizing outliers using Boxplots, Histograms, and Scatter plots.
- Understanding **Trimming** (removing outlier samples) vs **Capping** (Winsorization / setting upper & lower bounds).
- **Z-Score Method**: Identifying outliers in normally distributed (Gaussian) features.
- **IQR (Interquartile Range) Method**: Detecting outliers in skewed feature distributions.
- **Percentile / Quantile Method**: Setting absolute upper and lower percentile cutoffs (e.g., 1st and 99th percentiles).

---

## 📚 Topics Covered

Organized logically by statistical distribution and detection methodology:

### 🟢 Beginner
- **Outlier Fundamentals**: Definition of outliers, causes (measurement error, true rare events), and visual identification using box plots.
- **Trimming vs Capping**: Evaluating when to drop outlier rows versus capping extreme values.

### 🟡 Intermediate
- **Z-Score Method (Gaussian Data)**:
  - Calculation: $Z = \frac{X - \mu}{\sigma}$.
  - Identifying values where $|Z| > 3$ (outside 3 standard deviations from the mean).
  - Implementation on symmetrical features (`Example.ipynb`).

### 🔴 Advanced
- **IQR Method (Skewed Data)**:
  - Calculating Quartiles: $Q_1$ (25th percentile), $Q_3$ (75th percentile), $IQR = Q_3 - Q_1$.
  - Setting Bounds: $\text{Lower Bound} = Q_1 - 1.5 \times IQR$, $\text{Upper Bound} = Q_3 + 1.5 \times IQR$.
  - Implementation on placement datasets (`placement.csv`).
- **Percentile / Quantile Capping**: Capping extreme values at fixed percentiles (e.g., 1% and 99% limits) on height-weight datasets (`weight-height.csv`).

| Method | Subfolder / Files | Target Distribution | Level |
| :--- | :--- | :--- | :--- |
| **Z-Score Method** | `Outlier Detection and Removal using Z-score Method/` | Normally distributed (Gaussian) data | Intermediate |
| **IQR Method** | `Outlier Detection and Removal using the IQR Method/` | Skewed / non-Gaussian data | Intermediate → Advanced |
| **Percentile Method** | `Outlier Detection using the Percentile Method/` | Arbitrary continuous data | Advanced |

---

## 📂 File Structure

```text
12_Outliers in Machine Learning/
├── README.md
├── Notes.md
├── Outlier Detection and Removal using the IQR Method/
│   ├── Example.ipynb
│   └── placement.csv
├── Outlier Detection and Removal using Z-score Method/
│   ├── Example.csv
│   └── Example.ipynb
└── Outlier Detection using the Percentile Method/
    ├── Example.ipynb
    └── weight-height.csv
```
