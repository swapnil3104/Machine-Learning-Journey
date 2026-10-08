# 📄 05. Working with CSV Files

## 📌 Folder Overview
Welcome to the **CSV Handling** module! Comma-Separated Values (CSV) is the most ubiquitous file format for tabular data in data science. This folder covers methods for loading, parsing, cleaning, and processing CSV datasets using native Python modules as well as the Pandas library.

### Why is this topic important?
Almost every tabular dataset in machine learning originates from CSV files or relational database exports. Knowing how to efficiently read, parameterize, clean, and write CSV files ensures seamless data ingestion into machine learning pipelines without data corruption or encoding bugs.

### What You Will Learn
- Reading and writing CSV files using standard Python (`open`, `csv` module) and Pandas (`pd.read_csv`).
- Parameterizing `pd.read_csv()`: specifying delimiters, custom column headers, index columns, and selecting specific feature columns (`usecols`).
- Handling common file read issues: character encodings (UTF-8, Latin-1), header missingness, and malformed rows.
- Efficient memory management: processing large CSV files using chunking (`chunksize`).

---

## 📚 Topics Covered

Organized logically from standard file reading to advanced Pandas parameters:

### 🟢 Beginner
- **CSV Fundamentals**: Understanding plain-text comma-separated format and structure.
- **Standard File I/O**: Opening and reading CSV files in base Python.

### 🟡 Intermediate
- **Pandas Data Loading**: `pd.read_csv()` basics, setting custom column names, and assigning index columns.
- **Handling Delimiters & Encodings**: Dealing with tab-separated, semicolon-separated, and non-UTF8 files.

### 🔴 Advanced
- **Advanced Load Parameters**:
  - `usecols` for memory optimization during feature loading.
  - `dtype` specification for preventing type inference overhead.
  - `chunksize` for streaming large datasets that exceed RAM capacity.

| Concept | File / Notebook | Description | Level |
| :--- | :--- | :--- | :--- |
| **Base CSV Parsing** | `CSV.ipynb` | Fundamentals of reading CSVs in Python and Pandas. | Beginner |
| **Advanced CSV Load** | `working-with-csv.ipynb` | Comprehensive parameters (`sep`, `encoding`, `chunksize`, `usecols`). | Intermediate → Advanced |
| **Sample Dataset** | `student-data.csv` | Practice tabular dataset used for loading and parsing exercises. | Practice Data |

---

## 📂 File Structure

```text
05_CSV/
├── README.md
├── CSV.ipynb
├── student-data.csv
└── working-with-csv.ipynb
```
