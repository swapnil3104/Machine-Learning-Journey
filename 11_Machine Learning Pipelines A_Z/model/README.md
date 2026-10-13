# 📦 Serialized Pipeline Model Artifacts

## 📌 Folder Overview
This folder contains serialized Python `pickle` (`.pkl`) artifacts generated during the training phase of the Machine Learning Pipelines module. These pre-fitted transformers and model estimators are used for inference on unseen data.

### Artifact Descriptions
- `clf.pkl`: Serialized Scikit-Learn classification model fitted on preprocessed features.
- `ohe_sex.pkl`: Fitted `OneHotEncoder` transformer for encoding gender categories.
- `ohe_embarked.pkl`: Fitted `OneHotEncoder` transformer for encoding passenger embarkation ports.

---

## 📂 File Structure

```text
model/
├── README.md
├── clf.pkl
├── ohe_embarked.pkl
└── ohe_sex.pkl
```
