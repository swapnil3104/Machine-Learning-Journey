# 🎲 19. Naive Bayes Classifier

## 📌 Folder Overview
Welcome to the **Naive Bayes Classifier** module! Naive Bayes is a fast, probabilistic supervised learning algorithm grounded in Bayes' Theorem. This folder covers Bayes' probability math, the conditional independence assumption, Laplace (additive) smoothing, step-by-step manual calculations on the Play Tennis dataset, and Scikit-Learn implementations.

### Why is this topic important?
Despite its simple ("naive") assumption that features are conditionally independent given the class, Naive Bayes performs remarkably well in real-world applications. It is exceptionally fast, requires minimal training data, scales well to high dimensions, and serves as an industry standard for Text Classification, Spam Filtering, and Sentiment Analysis.

### What You Will Learn
- **Bayes' Theorem**:
  $$P(y | x_1, x_2, \dots, x_n) = \frac{P(y) \prod_{i=1}^{n} P(x_i | y)}{P(x_1, x_2, \dots, x_n)}$$
- **The "Naive" Assumption**: Why assuming feature conditional independence simplifies high-dimensional joint probability calculations.
- **Laplace Smoothing (Additive Smoothing)**: Preventing Zero-Probability problems ($P(x_i | y) = 0$) when evaluating unseen categorical features:
  $$P(x_i | y) = \frac{\text{count}(x_i, y) + \alpha}{\text{count}(y) + \alpha \cdot K}$$
- **Naive Bayes Variants**:
  - **Gaussian Naive Bayes**: Continuous features assumed to follow a normal distribution.
  - **Multinomial Naive Bayes**: Discrete word frequencies and count data (Text Mining).
  - **Bernoulli Naive Bayes**: Binary / boolean feature vectors.
- **Manual Calculation**: Step-by-step probability tables computed on `play_tennis.csv`.

---

## 📚 Topics Covered

Organized logically from probability fundamentals to algorithm variants:

### 🟢 Beginner
- **Probability Essentials**: Prior Probability $P(y)$, Likelihood $P(x|y)$, Marginal Probability $P(x)$, and Posterior Probability $P(y|x)$.
- **Play Tennis Dataset**: Predict whether to play tennis based on Outlook, Temperature, Humidity, and Wind.

### 🟡 Intermediate
- **The Naive Assumption**: Converting $P(x_1, x_2, \dots, x_n | y)$ into $\prod P(x_i | y)$.
- **Laplace Smoothing**: Resolving zero-frequency issues using smoothing parameter $\alpha = 1$.

### 🔴 Advanced
- **Algorithm Variants**:
  - **Gaussian Naive Bayes**: Estimating parameters using mean $\mu$ and variance $\sigma^2$.
  - **Multinomial & Bernoulli Naive Bayes**: Applying count-based probabilities for text classification.
- **Code from Scratch & Scikit-Learn**: Custom Python calculation vs `GaussianNB` / `MultinomialNB` classes.

| Task / File | Notebook / File | Focus Area | Level |
| :--- | :--- | :--- | :--- |
| **Manual Hand Calculations** | `Naive Bayes Classifier From basic - Play tennis dataset.ipynb` | Step-by-step Bayes probability table construction. | Beginner → Intermediate |
| **Algorithmic Fitting** | `Naive Bayes Classification using Algorithm — Play Tennis Dataset.ipynb` | Scikit-Learn Naive Bayes pipeline implementation. | Intermediate |
| **Play Tennis Datasets** | `play_tennis.csv`, `play_tennis Basic.csv` | Practice categorical weather datasets. | Practice Data |
| **Comprehensive Notes** | `Naive_Bayes_Classifier_in_Machine_Notes.md` | Theoretical guide, math derivations, & interview questions. | Reference |

---

## 📂 File Structure

```text
19_Naive Bayes Classifier/
├── README.md
├── Naive Bayes Classification using Algorithm — Play Tennis Dataset.ipynb
├── Naive Bayes Classifier From basic - Play tennis dataset.ipynb
├── Naive_Bayes_Classifier_in_Machine_Notes.md
├── play_tennis Basic.csv
└── play_tennis.csv
```
