# HW2-ML-2026AK05104
Assignment repository for ML (AIMLZG565) course

# Machine Learning Assignment 2

## Problem statement

Build and compare five classification models for a public dataset, evaluate each model using the required classification metrics, and present the results in an interactive Streamlit application. The application should accept a labeled test CSV, load the saved models, and display evaluation metrics and a confusion matrix or classification report without retraining.

## Dataset description

**Dataset:** [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online%2Bshoppers%2Bpurchasing%2Bintention%2Bdatase) from the UCI Machine Learning Repository.

The dataset contains 12,330 online shopping sessions and 17 input features. The binary target, `Revenue`, indicates whether a session resulted in a purchase. The classes are imbalanced: approximately 15.5% of sessions ended in a purchase. 

Categorical and numerical inputs will be preprocessed as appropriate for each model. Any preprocessing will be fitted on training data and saved with the corresponding model so the same transformations can be applied to uploaded test data.

## GitHub Repository Link

**Repository:** [HW2-ML-2026AK05104](https://github.com/radonium86-wq/HW2-ML-2026AK05104)

## Models used

The following five classifiers were trained and evaluated on the dataset. Below are the matrics evaluated for the models.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8816 | 0.8809 | 0.7394 | 0.3639 | 0.4877 | 0.4641 |
| Decision Tree | 0.8658 | 0.7378 | 0.5687 | 0.5524 | 0.5604 | 0.4813 |
| K-Nearest Neighbors (KNN) | 0.8508 | 0.6908 | 0.5530 | 0.1911 | 0.2840 | 0.2617 |
| Naive Bayes | 0.7328 | 0.7728 | 0.3249 | 0.6728 | 0.4382 | 0.3229 |
| Support Vector Machine (SVM) | 0.8832 | 0.8701 | 0.7554 | 0.3639 | 0.4912 | 0.4713 |

### Model performance observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong accuracy and AUC with good precision, but recall remains low, indicating many missed positive cases. |
| Decision Tree | Balanced performance with the best F1 and MCC among the models, though AUC is weaker than Logistic Regression and SVM. |
| KNN | Acceptable accuracy but weak recall and poor AUC, suggesting it struggles with the imbalanced dataset. |
| Naive Bayes | Highest recall but poor precision and lower overall accuracy, so it is less reliable for the positive class. |
| Support Vector Machine | Best accuracy and precision overall; strong general performance, though Logistic Regression slightly leads in AUC. |
| **Overall winner for this dataset** | **Support Vector Machine (SVM)** based on the highest accuracy and strong precision. |

