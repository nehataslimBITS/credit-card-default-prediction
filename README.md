# Credit Card Default Prediction

## Problem Statement

The objective of this project is to build and compare multiple machine learning classification models for predicting whether a credit card client will default on their payment in the next month.

The following machine learning models were implemented and evaluated:

1. Logistic Regression
2. Decision Tree
3. kNN
4. Naive Bayes
5. Random Forest

The models were evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and MCC.

---

## Dataset

The dataset used for this project is the Default of Credit Card Clients dataset.

The dataset contains 30,000 records and 23 input features used for prediction.

The target variable is:

- `default` — indicates whether the client defaulted on payment next month.

The dataset contains information related to credit limit, demographic information, repayment status, bill amounts, and previous payment amounts.

The data was divided into:

- Training data: 24,000 records
- Testing data: 6,000 records

The training/testing split used was 80/20.

---

## Models Used

### 1. Logistic Regression

Logistic Regression was used as a linear classification model for predicting credit card default.

### 2. Decision Tree

Decision Tree was used to model nonlinear relationships between the input features and the default outcome.

### 3. kNN

k-Nearest Neighbors was used as a distance-based classification model.

### 4. Naive Bayes

Naive Bayes was used as a probabilistic classification model.

### 5. Random Forest

Random Forest was used as an ensemble classification model consisting of multiple decision trees.

---

## Model Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8077 | 0.7076 | 0.6868 | 0.2396 | 0.3553 | 0.3244 |
| Decision Tree | 0.7145 | 0.6075 | 0.3694 | 0.4115 | 0.3893 | 0.2042 |
| kNN | 0.7928 | 0.7014 | 0.5487 | 0.3564 | 0.4322 | 0.3233 |
| Naive Bayes | 0.7525 | 0.7249 | 0.4515 | 0.5539 | 0.4975 | 0.3386 |
| Random Forest | 0.8120 | 0.7506 | 0.6325 | 0.3580 | 0.4572 | 0.3749 |

---

## Observations

### Logistic Regression

Logistic Regression achieved an accuracy of 0.8077 and an AUC of 0.7076. It achieved the highest precision among the five models, but its recall for the default class was relatively low.

### Decision Tree

Decision Tree achieved an accuracy of 0.7145 and an AUC of 0.6075. It had lower overall performance compared with the other models.

### kNN

kNN achieved an accuracy of 0.7928 and an AUC of 0.7014. Its F1 Score was 0.4322, providing balanced performance compared with Decision Tree.

### Naive Bayes

Naive Bayes achieved an accuracy of 0.7525 and an AUC of 0.7249. It achieved the highest recall of 0.5539 and the highest F1 Score of 0.4975 among the five models.

### Random Forest

Random Forest achieved the highest accuracy of 0.8120 and an AUC of 0.7506. It also achieved the highest MCC of 0.3749.

---

## Overall Winner

Based on the evaluation results, Random Forest achieved the best overall performance.

It obtained:

- Accuracy: 0.8120
- AUC: 0.7506
- Precision: 0.6325
- Recall: 0.3580
- F1 Score: 0.4572
- MCC: 0.3749

Random Forest achieved the highest Accuracy, AUC, and MCC among the evaluated models.

---

## Streamlit Application

The project includes a Streamlit application that allows users to:

- Select a machine learning model
- Upload a test CSV file
- Generate predictions
- View prediction results
- View evaluation metrics
- View the confusion matrix

---

## Project Structure

```text
2025Neha/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
├── model/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   └── scaler.pkl
│
└── ML_Assignment.ipynb