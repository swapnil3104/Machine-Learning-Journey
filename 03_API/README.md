# 🌐 03. Fetching Data via REST APIs

## 📌 Folder Overview
Welcome to the **API Fetching** module! Application Programming Interfaces (APIs) allow developers to retrieve live data directly from web platforms, external databases, and cloud web services. This folder demonstrates how to fetch structured data using Python's `requests` library and transform raw JSON responses into Pandas DataFrames.

### Why is this topic important?
In real-world data science and machine learning projects, datasets are rarely delivered solely as ready-to-use CSV files. Data engineers and ML engineers frequently fetch live data streams, external APIs (e.g., weather, financial, social media, TMDB movie databases), and web services to build automated training pipelines.

### What You Will Learn
- Fundamentals of HTTP communications (GET requests, query parameters, authorization headers).
- Using Python's `requests` library to communicate with web endpoints.
- Parsing JSON response payloads and inspecting HTTP response status codes (200, 404, 500).
- Iterating across paginated API endpoints to construct multi-page datasets for ML models.

---

## 📚 Topics Covered

Organized logically from HTTP basics to dataset construction:

### 🟢 Beginner
- **HTTP Protocol Basics**: Understanding GET requests, endpoints, and status codes.
- **Python `requests` Library**: `requests.get()`, URL formatting, and checking response status.

### 🟡 Intermediate
- **JSON Parsing**: Extracting response JSON via `.json()` and traversing key-value dictionaries.
- **Data Structuring**: Converting lists of JSON objects into Pandas DataFrames.

### 🔴 Advanced
- **Paginated API Fetching**: Writing loops to extract data across hundreds of API pages (e.g., TMDB top-rated movies) to assemble comprehensive tabular datasets.

| Concept | File / Notebook | Focus Area | Level |
| :--- | :--- | :--- | :--- |
| **API Fetching** | `API_Fatching.ipynb` | HTTP requests, parsing JSON API payloads, and building DataFrames. | Beginner → Advanced |

---

## 📂 File Structure

```text
03_API/
├── README.md
└── API_Fatching.ipynb
```
