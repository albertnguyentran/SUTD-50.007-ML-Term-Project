# ArXiv CS Paper Classification — Project Instructions

## Overview

Multi-class text classification task: given the abstract of a scientific paper, predict its primary Computer Science (CS) subcategory (40 classes) using the ArXiv dataset.

Teams: 4 or 5 people. Performance evaluated using **Macro F1 Score**.

---

## Task 1: Logistic Regression from Scratch

Implement binary Logistic Regression from scratch, then adapt it for multi-class classification.

**Restrictions:** Do NOT use `sklearn` logistic regression, `OneVsRestClassifier`, or any pre-defined logistic regression/multi-class wrappers.

### Required Functions

- `sigmoid(z)` — maps real input to (0, 1)
- `loss(y, y_hat)` — Log Loss function
- `gradients(X, y, y_hat)` — returns `dw` and `db` via gradient descent
- `train(X, y, bs, epochs, lr)` — training loop
- `predict(X)` — outputs binary class labels

### Multi-Class Adaptation

- Train one binary classifier per class (e.g., One-vs-Rest)
- Use `np.argmax()` over all binary classifiers' predicted probabilities to select final class

### Deliverables

- `1a`: Binary Logistic Regression implementation
- `1b`: Multi-class adaptation implementation
- `1c`: `LogReg_Prediction.csv` — predictions on test set, submitted to Kaggle

---

## Task 2: Dimensionality Reduction

Compare two approaches to reduce the 5000-feature TF-IDF space.

**Allowed:** `sklearn` packages for TF-IDF, PCA, TruncatedSVD, and KNN.

### Methods

- **Feature Selection:** Re-compute TF-IDF restricted to top N features
- **Dimension Reduction:** Apply PCA/TruncatedSVD to project 5000 features → N components

### Feature Sizes to Test

`N = 2000, 1000, 500, 100`

### Model

KNN with `n_neighbors=39`

### Deliverables

- `2a`: Code for both methods at all 4 feature sizes
- `2b`: Report Macro F1 scores on test set for all 8 configurations (submit to Kaggle)

---

## Task 3: Best Model — Race to the Top

Apply any non-deep-learning ML models to maximize classification performance.

**Restrictions:** No deep learning models.

### Bonus Marks (awarded after private leaderboard release, top 3 teams present in Week 13)

- 1 mark: 3rd highest score
- 2 marks: 2nd highest score
- 3 marks: 1st highest score

### Deliverables

- `3a`: Code for all models tried, with comments on model type and key hyperparameters
- `3b`: Kaggle submission under registered team name

---

## Task 4: Final Report (PDF)

Document your ML journey. Address:

1. Introduction to your best performing model (how it works)
2. Hyperparameter tuning process — what you tried and how you arrived at the best settings
3. Self-learned content beyond the course — what it was and whether it should be taught in future ML courses

### Deliverables

- `4a`: Final report in PDF

---

## Submission Format

CSV file with header:

```
id,label_id
173148,38
29098,28
```

---

## Evaluation Metric

**Macro F1 Score** — computes F1 per class and averages equally across all classes, ensuring niche categories are treated fairly.
