# ArXiv CS Paper Classification — Verbose Kaggle Instructions

## Description

This is a team project. You are encouraged to form teams in any way you like, but each team must consist of either 4 or 5 people.

## Project Summary

The rapid expansion of scientific literature makes it increasingly difficult for researchers to navigate new findings manually. The ArXiv repository, containing millions of preprints, requires robust automated systems to ensure papers are correctly routed to their respective scientific communities.

This project formulates this challenge as a multi-class text classification task:

> Given the abstract of a scientific paper, accurately predict its primary Computer Science (CS) subcategory.

---

## Task 1: Implement Logistic Regression from Scratch (10 marks)

Recall that you have learned about binary Logistic Regression in your earlier classes. Your task is to implement a foundational binary Logistic Regression model from scratch.

However, you will quickly notice a core problem: our ArXiv dataset is a multi-class classification task (with up to 40 distinct Computer Science subcategories), but standard Logistic Regression is strictly a binary classifier.

Therefore, you must first implement the base binary classifier, and then explore and implement an approach to adapt your binary model for multi-class classification (for example, by designing a One-vs-Rest (OvR) architecture).

**Important restriction:** You are **NOT TO USE** the `sklearn` logistic regression package, `OneVsRestClassifier`, or any other pre-defined logistic regression/multi-class wrappers for this task. Usage of any pre-defined packages for the core algorithm will result in **0 marks**.

### Key Task Deliverables

1. `1a`: Code implementation of the foundational binary Logistic Regression model.
2. `1b`: Code implementation of your chosen multi-class adaptation (how you used your binary model to predict the multiple classes).
3. `1c`: Prediction made by your multi-class Logistic Regression on the test set. Submit your predicted labels to Kaggle and include the final prediction output in your project submission labeled as `LogReg_Prediction.csv`.

### Tips & Implementation Guidelines

Your foundational binary implementation should include the following core functions:

- `sigmoid(z)`: A function that takes in a real number input and returns an output value between 0 and 1.
- `loss(y, y_hat)`: A loss function that allows us to minimize and determine the optimal parameters. The function takes in the actual labels `y` and the predicted labels `y_hat`, and returns the overall training loss. You should be using the Log Loss function taught in class.
- `gradients(X, y, y_hat)`: The Gradient Descent Algorithm to find the optimal values of our parameters. The function takes in the training feature `X`, actual labels `y`, and the predicted labels `y_hat`, and returns the partial derivative of the Loss function with respect to weights (`dw`) and bias (`db`).
- `train(X, y, bs, epochs, lr)`: The training function for your model.
- `predict(X)`: The prediction function to output binary class labels.

Scaling to Multi-Class:

- Once your base model works, consider how to loop through your dataset to train multiple binary classifiers (e.g., `cs.AI` vs `Not cs.AI`).
- You will need to write a custom prediction function that passes a test sample through all your trained binary models and uses `np.argmax()` to select the class with the highest predicted probability.

---

## Task 2: Apply Dimension Reduction Techniques (10 marks)

High-dimensional data can lead to slow training times, overfitting, and increased memory usage. The train dataset contains 5000 TF-IDF features.

In this task, you are to compare two distinct approaches for handling high-dimensional text data:

- dimensionality reduction via PCA (Principal Component Analysis), versus
- direct feature selection by simply restricting the number of top TF-IDF features used.

### Key Task Deliverables

1. `2a`: Code implementation to reduce the feature space to `2000`, `1000`, `500`, and `100` features using two separate methods:
   - **Feature Selection:** Re-computing the TF-IDF matrix while restricting the vocabulary to the top `N` features.
   - **Dimension Reduction:** Applying PCA (or its sparse equivalent) to project the original 5000 TF-IDF features down to `N` components.
     - Note: You are allowed to use the `sklearn` packages for both TF-IDF and PCA/TruncatedSVD for this task.
2. `2b`: Report the Macro F1 scores for applying `2000`, `1000`, `500`, and `100` features/components on the test set for both methods (8 scores in total).
   - You must submit your predicted labels to Kaggle to retrieve the Macro F1 scores for the test set and report the results in your final report.
   - Use KNN as the machine learning model for training and prediction with **`n_neighbors=2`**.
   - You are allowed to use the `sklearn` package for KNN implementation.

---

## Task 3: Try Other Machine Learning Models and Race to the Top! (25 marks)

In this course, you are exposed to many other machine learning models. For this task, you can apply any other machine learning models (taught in the course or not) to improve the ArXiv classification performance.

You are **NOT TO USE** any deep learning approach.

To make this task fun, there is a race to the top. Bonus marks will be awarded as follows:

- `1` mark: third-highest score on the private leaderboard
- `2` marks: second-highest score on the private leaderboard
- `3` marks: top-highest score on the private leaderboard

The private leaderboard will only be released after project submission. The top 3 teams will present their solution in week 13 to get the bonus marks.

### Key Task Deliverables

1. `3a`: Code implementation of all the models that you have tried. Include comments on your implementation (models used and key hyperparameter settings).
2. `3b`: Submit your predicted labels for the test set to Kaggle. You will be able to see your model performance on the public leaderboard. Make your submission under your registered team name, as points are awarded according to that ranking.

---

## Task 4: Documenting Your Journey and Thoughts (5 marks)

All good projects must come to an end. You need to document your machine learning journey in your final report.

Please include the following:

1. An introduction of your best performing model (how it works).
2. How you tuned the model. Discuss the parameters that you used and the different parameters that you tried before arriving at the best results.
3. Whether you self-learned anything beyond the course. If yes, what they are, and whether they should be taught in future Machine Learning courses.

### Key Task Deliverables

- `4a`: A final report (PDF) answering the above questions.

---

## Acknowledgements

This dataset is a subset from Kaggle arXiv dataset:  
[https://medium.com/@kaggleteam/leveraging-ml-to-fuel-new-discoveries-with-the-arxiv-dataset-981a95bfe365](https://medium.com/@kaggleteam/leveraging-ml-to-fuel-new-discoveries-with-the-arxiv-dataset-981a95bfe365)

---

## Evaluation

Model performance is assessed using the **Macro F-Score**.

This metric is critical because it treats all CS subcategories equally, ensuring that the model performs well on niche fields (like `cs.OS`) and not just on high-volume categories (like `cs.CV` or `cs.LG`).

---

## Submission File

For each `id` in the test set (representing a post), you must predict its class (`0` - `38`).

The file should contain a header and have the following format:

```text
id,label_id
173148,38
29098,28
28211,14
136101,7
```
