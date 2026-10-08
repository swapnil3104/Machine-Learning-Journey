# ⚡ 08. Automated EDA with Pandas Profiling

## 📌 Folder Overview
Welcome to the **Pandas Profiling** module! Pandas Profiling (now known as `ydata-profiling`) is a powerful Python package that automates the Exploratory Data Analysis process by generating comprehensive, interactive HTML statistical reports with a single line of code.

### Why is this topic important?
While manual EDA (writing custom Pandas queries and Seaborn plots) provides granular control, automated profiling tools save significant time during initial data inspection. In industry settings, generating a profiling report allows engineers to rapidly assess dataset quality, missing value ratios, correlations, and cardinality warnings within seconds.

### What You Will Learn
- Installing and setting up `ydata-profiling` / `pandas-profiling`.
- Generating full-spectrum `ProfileReport` objects from Pandas DataFrames.
- Exporting interactive HTML reports for business reporting and peer review.
- Interpreting automated warning alerts: high correlation flags, zero values, missing value matrices, and skewness indicators.

---

## 📚 Topics Covered

Organized logically from automated setup to report interpretation:

### 🟢 Beginner
- **Introduction to Automated EDA**: Comparing manual EDA vs automated profiling.
- **Loading Target Dataset**: Reading tabular CSV files (`Train.csv`).

### 🟡 Intermediate
- **Generating Profile Reports**:
  - `ProfileReport(df, title="...")` initialization.
  - Rendering reports inline inside Jupyter Notebooks (`.to_widgets()`, `.to_notebook_iframe()`).
  - Saving reports as standalone HTML files (`.to_file("report.html")`).

### 🔴 Advanced
- **Interpreting Advanced Report Sections**:
  - **Overview**: Dataset statistics, duplicate rows, total memory size.
  - **Variables**: Quantile statistics, descriptive metrics, histogram distributions.
  - **Correlations**: Pearson, Spearman, Kendall, and Phik ($\phi_k$) correlation matrices.
  - **Missing Values**: Bar charts, matrix visualizations, and dendrograms.

| File | Description | Level |
| :--- | :--- | :--- |
| `Pandas Profiling .ipynb` | Notebook demonstrating automated profile report generation on `Train.csv`. | Beginner → Intermediate |
| `Notes.md` | Theoretical reference notes on Pandas Profiling configuration and metric interpretation. | Reference |
| `Train.csv` | Sample tabular dataset used for profiling and automated EDA. | Practice Dataset |

---

## 📂 File Structure

```text
08_Pandas Profiling/
├── README.md
├── Notes.md
├── Pandas Profiling .ipynb
└── Train.csv
```
