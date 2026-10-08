# 📍 20. K-Nearest Neighbors (KNN) & Decision Surfaces

## 📌 Folder Overview
Welcome to the **K-Nearest Neighbors (KNN)** module! KNN is an intuitive, non-parametric, instance-based (lazy learning) algorithm used for classification and regression tasks. This folder covers spatial distance metrics, hyperparameter $K$ selection, feature scaling requirements, decision boundary visualization using MLxtend, and real-world application on the Breast Cancer Wisconsin diagnostic dataset.

### Why is this topic important?
KNN makes no explicit structural assumptions about the underlying data distribution. It predicts new query samples based on the majority label or average target value of its $K$ nearest neighbors in feature space. Understanding KNN highlights key ML principles such as distance metrics, spatial geometry, and the necessity of feature scaling.

### What You Will Learn
- **Lazy Learning Mechanism**: Why KNN skips explicit model training and stores training instances in memory.
- **Distance Metrics**:
  - **Euclidean Distance**: $d(p, q) = \sqrt{\sum (p_i - q_i)^2}$
  - **Manhattan Distance**: $d(p, q) = \sum |p_i - q_i|$
  - **Minkowski Distance**: General distance metric $d(p, q) = \left( \sum |p_i - q_i|^p \right)^{1/p}$
- **Hyperparameter $K$ Selection**: Impact of small $K$ (overfitting, high variance) vs large $K$ (underfitting, high bias).
- **Decision Surface Plotting**: Visualizing spatial decision regions and boundaries using `mlxtend.plotting.plot_decision_regions`.
- **Medical Diagnostic Application**: Diagnosing malignant vs benign tumors on the Breast Cancer Wisconsin dataset.

---

## 📚 Topics Covered

Organized logically from spatial distance concepts to boundary plotting:

### 🟢 Beginner
- **KNN Intuition**: Majority voting for classification, mean averaging for regression.
- **Importance of Feature Scaling**: Why unscaled features ruin distance-based algorithms.

### 🟡 Intermediate
- **Selecting Optimal $K$**: Cross-validation curve plotting to detect the elbow point.
- **Breast Cancer Diagnosis**: Training KNN classifiers on `Breast Cancer Wisconsin (Diagnostic) Data Set.csv`.

### 🔴 Advanced
- **Decision Surfaces with MLxtend**:
  - Plotting non-linear decision regions in 2D space.
  - Observing how decision boundaries smooth out as $K$ increases.
  - Evaluating distance weighting (`weights='uniform'` vs `weights='distance'`).

| Notebook / Document | Focus Area | Key Concepts | Level |
| :--- | :--- | :--- | :--- |
| `knn-on-breast-cancer-dataset.ipynb` | Medical dataset classification pipeline. | Preprocessing, scaling, KNN tuning, evaluation. | Intermediate |
| `Decision_Surface_and_MLxtend_Notes.md` | Decision region plotting guide. | MLxtend visualization & boundary analysis. | Intermediate → Advanced |
| `K_Nearest_Neighbors_Machine_Learning_Notes.md` | Theoretical reference manual. | Distance metrics, lazy learning, math derivations. | Reference |
| `Breast Cancer Wisconsin (Diagnostic) Data Set.csv` | Medical Diagnostic Dataset. | Real-world feature matrix for classification. | Practice Data |

---

## 📂 File Structure

```text
20_K Nearest Neighbors/
├── README.md
├── Breast Cancer Wisconsin (Diagnostic) Data Set.csv
├── Decision_Surface_and_MLxtend_Notes.md
├── K_Nearest_Neighbors_Machine_Learning_Notes.md
└── knn-on-breast-cancer-dataset.ipynb
```
