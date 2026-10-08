# 🔍 07. Exploratory Data Analysis (EDA)

## 📌 Folder Overview
Welcome to the **Exploratory Data Analysis (EDA)** module! EDA is the crucial initial phase in any data science workflow where you examine, visualize, and analyze datasets to uncover underlying patterns, spot anomalies, test hypotheses, and verify assumptions using summary statistics and graphical representations.

### Why is this topic important?
"Garbage in, garbage out" is the cardinal rule of Machine Learning. Before feeding data to algorithms, EDA helps you detect missing data, skewed distributions, outliers, feature correlations, and data leakage. Thorough EDA informs all subsequent decisions in feature engineering, scaling, and model selection.

### What You Will Learn
- **Univariate Analysis**: Analyzing single variables using frequency tables, histograms, KDE plots, box plots, and bar charts.
- **Bivariate Analysis**: Examining pairwise relationships (Numerical vs Numerical, Categorical vs Numerical, Categorical vs Categorical).
- **Multivariate Analysis**: Analyzing 3+ features simultaneously using heatmaps, correlation matrices, pair plots, and facet grids.
- **Seaborn Visualizations**: Leveraging Seaborn (`sns.histplot`, `sns.boxplot`, `sns.heatmap`, `sns.pairplot`, `sns.catplot`) for statistical plots.

---

## 📚 Topics Covered

Organized logically from single-variable summaries to complex multi-feature interactions:

### 🟢 Beginner
- **Data Profiling**: `.info()`, `.describe()`, checking shape, data types, and null values.
- **Univariate Analysis**:
  - **Categorical Data**: Countplots, pie charts, frequency distributions.
  - **Numerical Data**: Histograms, Kernel Density Estimation (KDE), skewness, box plots for outlier detection.

### 🟡 Intermediate
- **Bivariate Analysis**:
  - **Numerical vs Numerical**: Scatter plots, line plots, correlation coefficients ($r$).
  - **Categorical vs Numerical**: Bar plots, box plots, violin plots (e.g., Titanic passenger class vs age/fare).
  - **Categorical vs Categorical**: Crosstabs, stacked bar plots, heatmaps (e.g., Pclass vs Survived).

### 🔴 Advanced
- **Multivariate Analysis & Feature Interactions**:
  - Correlation matrices and Seaborn heatmaps.
  - Multi-variable pairplots and hue segmentation.
  - Detailed EDA workflow on real datasets (`train.csv` - Titanic dataset, `student-data.csv`).

| Topic / Notebook | Key Techniques Covered | Level |
| :--- | :--- | :--- |
| `Univariate Analysis.ipynb` | Distribution plots, boxplots, histograms, countplots. | Beginner |
| `Seaborn.ipynb` | Comprehensive Seaborn statistical visualization suite. | Intermediate |
| `Bivariate and Multivariate Analysis.ipynb` | Scatterplots, correlation heatmaps, pairplots, crosstabs. | Advanced |
| `Notes.md` | Theoretical notes and analytical frameworks for EDA. | Reference |

---

## 📂 File Structure

```text
07_EDA/
├── README.md
├── Bivariate and Multivariate Analysis.ipynb
├── Check.ipynb
├── Example.ipynb
├── Example2.ipynb
├── Notes.md
├── Seaborn.ipynb
├── Univariate Analysis.ipynb
├── student-data.csv
└── train.csv
```
