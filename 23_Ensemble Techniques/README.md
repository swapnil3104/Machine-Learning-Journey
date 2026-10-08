# 🌲🌲 23. Ensemble Learning Techniques

## 📌 Folder Overview
Welcome to the **Ensemble Techniques** module! Ensemble methods combine predictions from multiple individual machine learning models ("weak learners") to produce a single, superior predictive model. This folder covers the complete spectrum of ensemble paradigms: Simple Ensembling, Bagging, Boosting, Stacking, Voting, Random Forest, AdaBoost, Gradient Boosting (GBDT), XGBoost, LightGBM, and CatBoost.

### Why is this topic important?
Ensemble techniques dominate tabular data machine learning competitions (such as Kaggle) and real-world industrial deployments. By aggregating diverse models, ensemble methods systematically reduce both variance (overfitting) and bias (underfitting), achieving performance far beyond any single standalone model.

### What You Will Learn
- **Simple Ensembling**: Max Voting (Classification), Averaging, and Weighted Averaging (Regression).
- **Bagging (Bootstrap Aggregating)**: Training parallel base estimators on random bootstrap samples to reduce variance (e.g., Bagging Classifier, Random Forest).
- **Random Forest**: Combining Bootstrap sampling with the Random Subspace method (feature bagging) and calculating Out-Of-Bag (OOB) error.
- **Boosting**: Training base estimators sequentially where each new model focuses on correcting the errors/residuals of its predecessor.
  - **AdaBoost**: Re-weighting misclassified training instances.
  - **Gradient Boosting (GBDT)**: Minimizing arbitrary loss functions using gradient descent steps.
  - **Modern GBDT Frameworks**: **XGBoost** (extreme gradient boosting with regularization), **LightGBM** (leaf-wise tree growth), and **CatBoost** (native categorical feature handling).
- **Stacking & Blending**: Combining heterogeneous models using a secondary Meta-Learner.

---

## 📚 Topics Covered

Organized logically from basic aggregation to state-of-the-art gradient boosting frameworks:

### 🟢 Beginner
- **Wisdom of the Crowd**: Why combining weak models improves generalization.
- **Voting Classifiers**: Hard Voting vs Soft Voting (predicting probabilities).

### 🟡 Intermediate
- **Bagging & Random Forests**:
  - Bootstrap sampling with replacement.
  - Feature selection per split ($\sqrt{p}$ for classification, $p/3$ for regression).
  - Out-of-Bag (OOB) scoring as a built-in validation set.

### 🔴 Advanced
- **Boosting Algorithms**:
  - **AdaBoost**: Exponential loss minimization and sample weight adjustments.
  - **Gradient Boosting (GBDT)**: Residual fitting and learning rate shrinking.
  - **XGBoost, LightGBM, CatBoost**: Hardware-accelerated, highly regularized modern gradient boosting libraries.
- **Stacking & Meta-Learning**: Multi-level architecture combining diverse base classifiers via Logistic Regression / Ridge meta-learners.

| Ensemble Family | Algorithms Covered | Primary Mechanism | Primary Benefit |
| :--- | :--- | :--- | :--- |
| **Voting / Averaging** | Hard Voting, Soft Voting, Weighted Mean | Direct prediction aggregation | Reduces variance |
| **Bagging** | Random Forest, Extra Trees, BaggingClassifier | Parallel bootstrap + feature sampling | Drastically reduces variance |
| **Boosting** | AdaBoost, GBDT, XGBoost, LightGBM, CatBoost | Sequential error / residual correction | Reduces bias and variance |
| **Stacking** | Multi-layer Stacking, Blending | Meta-learner trained on base predictions | Maximizes predictive accuracy |

---

## 📂 File Structure

```text
23_Ensemble Techniques/
├── README.md
└── Ensemble_Techniques_in_Machine_Learning.md
```
