# 📏 18. Model Evaluation & Performance Metrics

## 📌 Folder Overview
Welcome to the **Model Evaluation** module! Model Evaluation is the process of quantitatively assessing the predictive performance, reliability, and generalization capability of machine learning models. This folder covers confusion matrices, binary classification metrics, multiclass averaging strategies (Micro, Macro, Weighted), ROC-AUC curves, and real-world implementations on Iris and MNIST datasets.

### Why is this topic important?
Relying solely on Accuracy can be disastrous—especially on imbalanced datasets (e.g., fraud detection where 99.9% of samples are non-fraudulent). Choosing the right metric (Precision, Recall, F1-Score, ROC-AUC, or Log Loss) ensures that your model aligns with actual business objectives and risk tolerances.

### What You Will Learn
- **Confusion Matrix**: Identifying True Positives (TP), True Negatives (TN), False Positives (FP - Type I Error), and False Negatives (FN - Type II Error).
- **Classification Metrics**:
  - **Accuracy**: Overall proportion of correct predictions.
  - **Precision**: $\frac{\text{TP}}{\text{TP} + \text{FP}}$ (minimizing false alarms).
  - **Recall / Sensitivity**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$ (minimizing missed detections).
  - **F1-Score**: Harmonic mean of Precision and Recall: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$.
- **Multiclass Averaging Strategies**: Macro-averaging (unweighted class average), Micro-averaging (global sample aggregation), and Weighted-averaging (class support weighting).
- **Practical Multi-Class Evaluation**: Performance metrics computed on Iris species and MNIST handwritten digit recognition.

---

## 📚 Topics Covered

Organized logically from confusion matrix fundamentals to multiclass evaluation:

### 🟢 Beginner
- **Why Accuracy Fails**: The Imbalanced Classification Trap.
- **Confusion Matrix Structure**: Reading 2x2 and NxN contingency tables.

### 🟡 Intermediate
- **Precision vs Recall Trade-Off**:
  - High Precision priority: Spam detection, video recommendation.
  - High Recall priority: Cancer screening, credit card fraud detection.
- **F1-Score**: Balancing precision and recall into a single metric.

### 🔴 Advanced
- **Multiclass Metric Aggregation**:
  - **Macro Average**: Treats all classes equally regardless of frequency.
  - **Micro Average**: Aggregates total TP, FP, FN across all classes.
  - **Weighted Average**: Weights metrics by class support size.
- **Benchmark Evaluations**: Testing classifiers on Iris multi-class (`classification-metrics-multi-iris1.ipynb`) and MNIST digit recognition (`classification-metrics-multi-mnist1.ipynb`).

| Metric / Evaluation | Notebook / Notes File | Use Case | Level |
| :--- | :--- | :--- | :--- |
| **Binary Classification** | `classification-metrics-binary.ipynb` | Confusion matrix, Precision, Recall, F1, ROC-AUC. | Beginner → Intermediate |
| **Iris Multi-Class** | `classification-metrics-multi-iris1.ipynb` | Multi-class metric calculations on 3-class target. | Intermediate |
| **MNIST Multi-Class** | `classification-metrics-multi-mnist1.ipynb` | Evaluating 10-class handwritten digit recognition. | Advanced |
| **Detailed Notes** | `Model_Evaluation_machine_learning.md`, `Multiclass Precision, Recall & F1-Score.md` | Theoretical reference manual for evaluation metrics. | Reference |

---

## 📂 File Structure

```text
18_ Model Evaluation in Machine Learning/
├── README.md
├── Model_Evaluation_machine_learning.md
├── Multiclass Precision, Recall & F1-Score.md
├── classification-metrics-binary.ipynb
├── classification-metrics-multi-iris1.ipynb
└── classification-metrics-multi-mnist1.ipynb
```
