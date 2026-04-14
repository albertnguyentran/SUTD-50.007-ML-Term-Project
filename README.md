# ArXiv CS Paper Classification

A multi-class text classification project for SUTD 50.007 Machine Learning. Given the abstract of a scientific paper, the goal is to predict its primary Computer Science (CS) subcategory (40 classes) from the ArXiv dataset.

Model performance is evaluated using **Macro F1 Score** on Kaggle.

---

## Project Structure

```
SUTD-50.007-ML-Term-Project/
├── dataset/              # Local only — not tracked by git (see setup below)
├── instructions.md       # Full task instructions
├── rubric.md             # Grading rubric
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd SUTD-50.007-ML-Term-Project
```

### 2. Add the dataset

The `dataset/` folder is not tracked by git. You must add it manually.

Create a `dataset/` folder in the project root and place the following 4 files inside:

```
dataset/
├── train.csv
├── test.csv
├── label_taxonomy.csv
└── sample_submission_random_guess.csv
```

---

## Tasks

| Task | Description | Marks |
|------|-------------|-------|
| Task 1 | Logistic Regression from scratch (binary + multi-class) | 10 |
| Task 2 | Dimensionality reduction — PCA vs TF-IDF feature selection | 10 |
| Task 3 | Best ML model — race to the top on Kaggle | 25 (+3 bonus) |
| Task 4 | Final report | 5 |

See `instructions.md` for full details and `rubric.md` for grading criteria.

---

## Kaggle Submissions

Submit prediction CSVs to Kaggle under your registered team name. The submission format is:

```
id,label_id
173148,38
29098,28
```
