# ⚙️ 11. Machine Learning Pipelines A to Z

## 📌 Folder Overview
Welcome to the **Machine Learning Pipelines** module! Scikit-Learn `Pipeline` and `ColumnTransformer` objects allow you to chain feature preprocessing steps, encodings, imputations, and predictive models into a single, cohesive workflow. This folder compares workflows built with vs without pipelines and demonstrates how to serialize pipelines into production artifacts (`.pkl`).

### Why is this topic important?
Without pipelines, machine learning workflows often suffer from **Data Leakage** (e.g., fitting scalers on the entire dataset instead of train-only splits), messy redundant code, and complex deployment scripts. Pipelines enforce clean encapsulation, streamline cross-validation, and allow end-to-end inference on raw data using a single `pipeline.predict()` call.

### What You Will Learn
- Building ML workflows **with vs without pipelines** to understand code readability and data leakage prevention.
- Combining `SimpleImputer`, `OneHotEncoder`, `StandardScaler`, and Classifier models inside Scikit-Learn `Pipeline`.
- Integrating `ColumnTransformer` within Pipelines to route different feature subsets (numeric vs categorical) through distinct processing branches.
- Exporting fully trained pipelines using `pickle` (`pipe.pkl`, `clf.pkl`) and serving predictions on unseen test inputs.

---

## 📚 Topics Covered

Organized logically from raw scripts to production-ready serialized pipelines:

### 🟢 Beginner
- **Problems with Manual Workflows**: Data leakage risks, code duplication, and manual feature transformation tracking.
- **Pipeline Syntax**: Initializing `Pipeline([('imputer', ...), ('scaler', ...), ('model', ...)])`.

### 🟡 Intermediate
- **Combining ColumnTransformer & Pipeline**: Routing numerical columns through scaling and categorical columns through One-Hot Encoding inside a unified pipeline.
- **Comparing Models**: Training Titanic classification models with and without pipelines on `train.csv`.

### 🔴 Advanced
- **Pipeline Serialization & Production Deployment**:
  - Saving fitted pipeline objects using `pickle.dump(pipe, open('pipe.pkl', 'wb'))`.
  - Loading serialized pipeline artifacts (`predict-using-pipeline.ipynb`).
  - Executing end-to-end predictions directly on raw, un-preprocessed input JSON/DataFrames.

| Topic / Task | Notebook / File | Description | Level |
| :--- | :--- | :--- | :--- |
| **Workflow Without Pipeline** | `titanic-without-using-pipeline.ipynb` | Manual feature preprocessing and training risks. | Beginner |
| **Workflow With Pipeline** | `titanic-using-pipeline.ipynb` | Clean Scikit-Learn pipeline encapsulation. | Intermediate |
| **Prediction Inference** | `predict-using-pipeline.ipynb`, `predict-without-pipeline.ipynb` | Loading `.pkl` artifacts for inference. | Intermediate → Advanced |
| **Serialized Artifacts** | `pipe.pkl`, `model/clf.pkl`, `model/ohe_sex.pkl` | Pickled pipeline and model weights for deployment. | Production |

---

## 📂 File Structure

```text
11_Machine Learning Pipelines A_Z/
├── README.md
├── Notes.md
├── pipe.pkl
├── predict-using-pipeline.ipynb
├── predict-without-pipeline.ipynb
├── readme.md
├── titanic-using-pipeline.ipynb
├── titanic-without-using-pipeline.ipynb
├── train.csv
└── model/
    ├── clf.pkl
    ├── ohe_embarked.pkl
    ├── ohe_sex.pkl
    └── readme.md
```
