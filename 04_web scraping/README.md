# 🕷️ 04. Web Scraping with Python

## 📌 Folder Overview
Welcome to the **Web Scraping** module! Web scraping is the technique of automatically extracting data from websites. This folder focuses on fetching HTML content from web pages using `requests` and parsing data using HTML parsers like `BeautifulSoup` to build custom tabular datasets.

### Why is this topic important?
When APIs or pre-built datasets are unavailable, web scraping empowers data scientists to gather unique, real-time web data (such as company ratings, job descriptions, product listings, or news articles) directly from web pages for analysis and machine learning modeling.

### What You Will Learn
- Inspecting web pages and understanding the HTML DOM structure (tags, classes, IDs).
- Sending HTTP requests with custom headers (`User-Agent`) to avoid anti-scraping blocks.
- Parsing HTML elements using BeautifulSoup (`find`, `find_all`, CSS selectors).
- Extracting text content, handling pagination across web pages, and saving scraped datasets into CSV format (e.g., AmbitionBox company data).

---

## 📚 Topics Covered

Organized logically from DOM inspection to automated web harvesting:

### 🟢 Beginner
- **HTML DOM Structure**: HTML tags (`<div>`, `<span>`, `<h1>`, `<a>`), attributes, and CSS classes.
- **HTTP Requests with Headers**: Standardizing HTTP requests and mimicking browser headers.

### 🟡 Intermediate
- **Parsing with BeautifulSoup**: Navigating the parse tree, using `soup.find_all()`, and extracting text attributes.
- **Single-Page Data Extraction**: Isolating company names, ratings, reviews, and domain details.

### 🔴 Advanced
- **Multi-Page Scraping & Export**: Looping over pagination URLs, handling edge cases/missing elements gracefully, and exporting clean tabular data (`ambitionbox_companies.csv`).

| Task | File / Notebook | Focus Area | Level |
| :--- | :--- | :--- | :--- |
| **Web Scraping Basics** | `web.ipynb` | HTML parsing, tags selection, and BeautifulSoup basics. | Beginner |
| **Scraping Practice** | `Example.ipynb` | Extracting data elements and manipulating text content. | Intermediate |
| **Company Data Scraper** | `web Scraping.ipynb` | Scraping AmbitionBox multi-page corporate ratings dataset. | Advanced |
| **Scraped Output Data** | `ambitionbox_companies.csv` | Exported CSV dataset containing scraped corporate records. | Intermediate |

---

## 📂 File Structure

```text
04_web scraping/
├── README.md
├── Example.ipynb
├── ambitionbox_companies.csv
├── web Scraping.ipynb
└── web.ipynb
```
