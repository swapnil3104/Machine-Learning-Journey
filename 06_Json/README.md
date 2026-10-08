# 📦 06. JSON Data Processing

## 📌 Folder Overview
Welcome to the **JSON Data Processing** module! JavaScript Object Notation (JSON) is the universal standard for semi-structured data exchange across APIs, modern databases, and web services. This folder covers loading, parsing, flattening, and normalizing JSON files in Python and Pandas.

### Why is this topic important?
Unlike flat tabular datasets, real-world data often comes in semi-structured, nested JSON formats (such as Kaggle recipe datasets, NoSQL database dumps, or REST API payloads). Machine learning models require tabular input matrices ($X$ features and $y$ targets); mastering JSON processing allows you to convert complex nested objects into clean Pandas DataFrames.

### What You Will Learn
- Python's standard `json` module: serializing (`json.dumps`) and deserializing (`json.loads`, `json.load`).
- Reading `.json` files directly into Pandas DataFrames using `pd.read_json()`.
- Flattening deeply nested JSON arrays and key-value objects using `pd.json_normalize()`.
- Working with complex real-world datasets like recipe classifications (`train.json`, `test.json`).

---

## 📚 Topics Covered

Organized logically from JSON syntax to nested array normalization:

### 🟢 Beginner
- **JSON Syntax & Structure**: Key-value pairs, nested dictionaries, arrays/lists, and data types.
- **Python `json` Library**: `json.load()` and `json.dumps()` operations.

### 🟡 Intermediate
- **Pandas Integration**: `pd.read_json()` for flat and semi-structured datasets.
- **Recipe Dataset Processing**: Loading multi-ingredient culinary datasets (`train.json`).

### 🔴 Advanced
- **Nested Structure Normalization**:
  - Unnesting dictionary columns with `pd.json_normalize()`.
  - Extracting lists into feature vectors for machine learning classification models.

| File / Dataset | Type | Description | Level |
| :--- | :--- | :--- | :--- |
| `json.ipynb` | Notebook | Core JSON parsing, loading, and Pandas DataFrame integration. | Beginner → Intermediate |
| `Example.ipynb` | Notebook | Hands-on exercises on normalizing nested JSON payloads. | Intermediate → Advanced |
| `train.json` | Dataset | Training dataset containing nested recipe ingredients and cuisines. | Real-world Data |
| `test.json` | Dataset | Evaluation dataset for recipe classification tasks. | Real-world Data |

---

## 📂 File Structure

```text
06_Json/
├── README.md
├── Example.ipynb
├── json.ipynb
├── test.json
└── train.json
```
