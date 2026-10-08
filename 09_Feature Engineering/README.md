# 🛠️ 09. Feature Engineering & Dimensionality Reduction

## 📌 Folder Overview
Welcome to the **Feature Engineering** module! Feature Engineering is the art and science of transforming raw data into informative, high-quality features that enhance machine learning model performance, stability, and interpretability. This folder covers the complete pipeline of feature transformation, encoding, scaling, mathematical transformations, date-time feature extraction, and dimensionality reduction via Principal Component Analysis (PCA).

### Why is this topic important?
Algorithms are only as good as the features provided to them. Raw data often contains skewed distributions, categorical text, mixed data types, unscaled numerical ranges, or redundant dimensions. Proper feature engineering improves model convergence speeds, prevents feature dominance, resolves non-normality, and combats the Curse of Dimensionality.

### What You Will Learn
- **Feature Scaling**: Normalization (MinMaxScaler) vs Standardization (StandardScaler).
- **Categorical & Numerical Encoding**: One-Hot Encoding, Ordinal Encoding, Binarization, and Binning.
- **Mathematical Transformations**: Function Transformations (Log, Reciprocal, Square Root) and Power Transformations (Box-Cox, Yeo-Johnson).
- **Pipeline Integration**: Using Scikit-Learn's `ColumnTransformer` to combine heterogeneous preprocessing steps.
- **Specialized Feature Handling**: Parsing Date/Time variables and handling mixed numerical-categorical strings.
- **Dimensionality Reduction**: Principal Component Analysis (PCA) step-by-step to compress high-dimensional feature spaces.

---

## 📚 Topics Covered

Organized logically from scaling basics to advanced dimensionality reduction:

### 🟢 Beginner
- **Feature Scaling**:
  - **Normalization**: Rescaling features to $[0, 1]$ interval using `MinMaxScaler`.
  - **Standardization**: Centering features to mean $\mu=0$ and standard deviation $\sigma=1$ using `StandardScaler`.
- **Date & Time Variables**: Extracting year, month, day, day of week, hour, and elapsed time features.

### 🟡 Intermediate
- **Categorical Data Encoding**:
  - **One-Hot Encoding (OHE)**: Handling dummy variable trap, $K-1$ encoding, and handling rare/unseen categories.
  - **Ordinal Encoding**: Mapping ordered categorical labels to discrete integers.
- **Numerical Discretization**: Binning continuous features into discrete interval bins (Uniform, Quantile, K-Means).
- **Handling Mixed Data**: Extracting numeric and string components from combined alphanumeric columns (e.g., Ticket codes, Cabin numbers).

### 🔴 Advanced
- **Mathematical Transformations**:
  - Log, Square Root, and Reciprocal transforms to convert skewed features into Gaussian normal distributions.
  - Power Transforms: Box-Cox (for strictly positive data) and Yeo-Johnson transforms.
- **ColumnTransformer**: Structuring clean, unified transformation workflows across varied column types.
- **Dimensionality Reduction (PCA)**:
  - Theoretical foundations of the Curse of Dimensionality.
  - Variance maximization, Covariance matrices, Eigenvalues, and Eigenvectors.
  - Step-by-step PCA calculation and variance ratio interpretation.

| Subfolder / Topic | Important Notebooks / Files | Description | Level |
| :--- | :--- | :--- | :--- |
| **01 Normalization** | `Normalization.ipynb`, `wine_data.csv` | MinMax scaling implementation and visual impact. | Beginner |
| **02 Standardization** | `Standardization.ipynb`, `Social_Network_Ads.csv` | Z-score standardization and algorithms sensitivity. | Beginner |
| **03 Encoding** | `one-hot-encoding.ipynb`, `Ordinal Encoding.ipynb`, `binarization.ipynb` | Encoding nominal, ordinal, and continuous data. | Intermediate |
| **04 Transformations** | `Column Transformer.ipynb`, `Function_Transformation_California_Housing.ipynb` | ColumnTransformer, Log, Box-Cox, Yeo-Johnson. | Intermediate → Advanced |
| **05 Mixed Data** | `Mixed Data Handling Example.ipynb` | Splitting alphanumeric mixed columns on Titanic data. | Intermediate |
| **06 Date & Time** | `working-with-dates-and-time.ipynb` | Parsing dates, extracting cycles, elapsed time features. | Intermediate |
| **07 Feature Construction** | `Example.ipynb` | Creating derived domain features (e.g., Family Size). | Intermediate |
| **08 PCA** | `pca_step_by_step (1).ipynb`, `Notes.md` | Principal Component Analysis and dimensionality reduction. | Advanced |

---

## 📂 File Structure

```text
09_Feature Engineering/
├── README.md
├── Notes.md
├── 01_Feature Scaling - Normalization/
│   ├── Normalization.ipynb
│   ├── Normalization_Notes.ipynb
│   └── wine_data.csv
├── 02_Feature Scaling - Standardization/
│   ├── Social_Network_Ads.csv
│   ├── Standardization.ipynb
│   └── Standardization_Notes.ipynb
├── 03_Encoding/
│   ├── Encoding Categorical Data/
│   │   ├── Notes.md
│   │   ├── One Hot Encoding/
│   │   │   ├── cars.csv
│   │   │   └── one-hot-encoding.ipynb
│   │   └── Ordinal Encoding/
│   │       ├── Ordinal Encoding.ipynb
│   │       └── customer.csv
│   └── Encoding Numerical Data/
│       ├── Notes.md
│       └── Binning/
│           ├── Example.ipynb
│           ├── Notes.md
│           ├── binarization.ipynb
│           └── train.csv
├── 04_Feature Transformtion/
│   ├── 01_Column Transformer in Machine Learning/
│   │   ├── Column Transformer.ipynb
│   │   ├── Notes.md
│   │   └── covid_toy.csv
│   └── 02_Mathematic Transformation/
│       ├── Notes.md
│       ├── Funtion Transform/
│       │   ├── Example.ipynb
│       │   ├── Function_Transformation_California_Housing.ipynb
│       │   ├── california_housing.csv
│       │   └── train.csv
│       └── Power Transform/
│           ├── Example.ipynb
│           └── concrete_data.csv
├── 05_Mixed Data/
│   ├── Mixed Data Handling Example.ipynb
│   └── titanic.csv
├── 06_Handling Date and Time Variables/
│   ├── messages.csv
│   ├── orders.csv
│   └── working-with-dates-and-time.ipynb
├── 07_Feature Construction/
│   ├── Example.ipynb
│   └── train.csv
└── 08_Curse of Dimensionality/
    ├── Notes.md
    └── Feature Extration/
        └── Principle Component Analysis/
            ├── Notes.md
            └── pca_step_by_step (1).ipynb
```
