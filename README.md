# CHURNIQ – Customer Churn Prediction & Retention Analytics

An end-to-end Machine Learning application that predicts customer churn, identifies high-risk customers, and provides data-driven retention insights through an interactive Streamlit dashboard.

## Project Overview

Customer churn is a major challenge for subscription-based businesses. CHURNIQ uses customer demographics, service subscriptions, contract details, and billing information to predict the likelihood of a customer leaving a service.

The project combines Exploratory Data Analysis (EDA), machine learning, model evaluation, explainable AI, and an interactive prediction application.

## Business Objectives

- Identify customers who are likely to churn.
- Understand customer behavior and churn patterns.
- Compare multiple machine learning algorithms.
- Prioritize customers based on predicted churn risk.
- Provide actionable customer retention recommendations.

## Dataset

- **Dataset:** IBM Telco Customer Churn
- **Records:** 7,043 customers
- **Features:** 21 original columns
- **Target Variable:** Churn
- **Problem Type:** Binary Classification

## Technology Stack

- **Programming:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost
- **Explainable AI:** SHAP
- **Model Persistence:** Joblib
- **Application:** Streamlit
- **Development:** Jupyter Notebook, VS Code

## Exploratory Data Analysis

The analysis explored customer demographics, contract types, tenure, internet services, payment methods, monthly charges, and churn distribution.

### Key Findings

- Overall customer churn rate: **26.54%**
- Month-to-month contract churn: **42.71%**
- One-year contract churn: **11.27%**
- Two-year contract churn: **2.83%**
- Customers who churned had an average tenure of approximately **18 months**.
- Customers who stayed had an average tenure of approximately **38 months**.
- Electronic check users had a churn rate of approximately **45.29%**.

These findings represent observed associations in the dataset, not proof of causation.

## Machine Learning Models

Three classification algorithms were evaluated:

| Metric | Logistic Regression | Random Forest | XGBoost |
|---|---:|---:|---:|
| Accuracy | 73.81% | 76.01% | 79.99% |
| Precision | 50.43% | 53.47% | 65.54% |
| Recall | 78.34% | 74.06% | 51.87% |
| F1 Score | 61.36% | 62.11% | 57.91% |
| ROC-AUC | 84.16% | 83.72% | 84.41% |

### Final Model

Logistic Regression was selected as the final model after considering the importance of identifying