# 🌳 22. Decision Trees & Advanced Tree Visualizations

## 📌 Folder Overview
Welcome to the **Decision Tree** module! Decision Trees are non-parametric supervised learning models used for both classification and regression tasks. This folder covers tree structure anatomy, node splitting metrics (Entropy, Information Gain, Gini Impurity, Variance Reduction), cost-complexity pruning, regression trees, and advanced interactive tree visualizations using the `dtreeviz` library.

### Why is this topic important?
Decision Trees closely mirror human decision-making logic, making them highly interpretable. Furthermore, Decision Trees form the fundamental structural building block for powerful ensemble algorithms (such as Random Forests, Gradient Boosting, XGBoost, LightGBM, and CatBoost).

### What You Will Learn
- **Tree Anatomy**: Root node, internal decision nodes, branches, and leaf (terminal) nodes.
- **Splitting Metrics for Classification**:
  - **Entropy**: Measure of impurity $H(S) = -\sum p_i \log_2(p_i)$.
  - **Information Gain**: Reduction in entropy after splitting $\text{IG}(S, A) = H(S) - \sum \frac{|S_v|}{|S|} H(S_v)$.
  - **Gini Impurity**: $\text{Gini}(S) = 1 - \sum p_i^2$.
- **Regression Trees**: Splitting continuous targets by minimizing Mean Squared Error (MSE) / Variance Reduction.
- **Overfitting & Pruning**: Controlling tree depth (`max_depth`), leaf size (`min_samples_leaf`), and cost-complexity pruning ($\alpha$).
- **Advanced Tree Visualization**: Rendering rich, interactive decision node graphics using `dtreeviz`.

---

## 📚 Topics Covered

Organized logically from tree splitting math to regression trees and visual rendering:

### 🟢 Beginner
- **Tree Building Intuition**: Splitting features recursively to maximize purity.
- **Classification Decision Trees**: Training decision trees on Social Network Ads and Breast Cancer datasets.

### 🟡 Intermediate
- **Splitting Math**: Hand-calculating Gini Impurity vs Information Gain.
- **Regression Trees**: Predicting continuous house values on the California Housing dataset (`Regression Tree/California Housing.ipynb`).

### 🔴 Advanced
- **Tree Pruning**: Preventing infinite tree depth through pre-pruning and post-pruning parameters.
- **Advanced Visualizations with `dtreeviz`**: Generating decision boundary overlays, feature split distributions, and node path predictions (`dtreeviz library/dtreeviz_demo.ipynb`).

| Module / Topic | Subfolder / File | Key Focus Area | Level |
| :--- | :--- | :--- | :--- |
| **Classification Demo** | `Decision Tree Classification Demo.ipynb`, `DT-on Breast Cancer.ipynb` | Classification trees, Gini vs Entropy. | Beginner → Intermediate |
| **Regression Trees** | `Regression Tree/` | Continuous target splitting, variance reduction. | Intermediate → Advanced |
| **`dtreeviz` Library** | `dtreeviz library/dtreeviz_demo.ipynb` | High-impact interactive tree visualizations. | Advanced |
| **Comprehensive Notes** | `Decision_Tree_in_Machine_Learning.md` | Theory, splitting math, pruning, & interview QA. | Reference |

---

## 📂 File Structure

```text
22_Decision Tree/
├── README.md
├── Breast Cancer Wisconsin (Diagnostic) Data Set.csv
├── DT-on Breast Cancer.ipynb
├── Decision Tree Classification Demo.ipynb
├── Decision_Tree_in_Machine_Learning.md
├── Social_Network_Ads(1).csv
├── dtreeviz library/
│   └── dtreeviz_demo.ipynb
└── Regression Tree/
    ├── California Housing.ipynb
    ├── Regression_Tree_in_Machine_Learning.md
    └── regression_tree_example(1).ipynb
```
