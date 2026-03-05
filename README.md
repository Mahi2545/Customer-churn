# Customer Churn Prediction

## 📊 Overview
End-to-end classification project to predict customer churn using machine learning on a real-world e-commerce/DTH customer dataset (11,260 records, 17 features after preprocessing). The project identifies at-risk customers and provides actionable retention strategies.

## 🎯 Objective
Build a predictive model to identify customers likely to churn, enabling proactive retention efforts and business strategy optimization based on Accuracy, F1 Score, Recall, Precision and AUC score.

## 🛠️ Technologies Used
- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Scikit-learn** - ML algorithms (LR, LDA, KNN, Naive Bayes, Bagging, AdaBoost, Gradient Boosting, SVM)
- **Matplotlib/Seaborn** - Visualizations
- **Jupyter Notebook** - Analysis environment

## 🔍 Key Features
- **EDA**: Comprehensive analysis of churn drivers — city tier, tenure, support interactions, payment preferences, account segment
- **Data Cleaning**: Special character treatment (['$','@','&','#','*'] replaced with NaN), missing value imputation, columns >30% missing dropped, category standardization ('Super +' → 'Super Plus', 'M/F' → Male/Female)
- **Class Imbalance**: Handled with **SMOTE** (balanced dataset: 13,112 resampled records) and 70:30 stratified train-test split
- **Multiple Models**: 8 classifiers evaluated across default, GridSearchCV (CV), and SMOTE-balanced (SM) variants
- **ROC-AUC Analysis**: Full ROC curve plotted and AUC computed for all models
- **Business Segmentation**: Customer Loyalty vs Spending quadrant analysis; city-tier based churn mapping

## 📈 Key Predictors
- Tenure, Monthly Revenue
- Account segment (Regular, Regular Plus, Super, Super Plus)
- City tier (Tier 1, 2, 3)
- Customer support call frequency, Complaint status
- Payment method (UPI, e-wallet, bank transfer)
- Marital status, Gender
- Coupon usage, Cashback amount

## 📈 Model Performance — Full Comparison (Table 9 from Final Report)

> CV = GridSearchCV tuned | SM = SMOTE balanced dataset

| Model | Train Accuracy | Test Accuracy | Test AUC Score |
|---|---|---|---|
| Logistic Regression | 83.91% | 83.98% | 0.75 |
| Logistic Regression - CV | 83.92% | 83.94% | 0.752 |
| Logistic Regression - SM | 68.21% | 67.87% | 0.748 |
| LDA | 84.21% | 83.62% | 0.748 |
| LDA - CV | 84.16% | 83.54% | 0.747 |
| LDA - SM | 68.60% | 67.83% | 0.748 |
| **KNN (Default — Best Model)** | **85.72%** | **84.04%** | **0.715** |
| KNN - CV | 85.82% | 84.16% | 0.72 |
| KNN - SM | 71.38% | 66.69% | 0.736 |
| Naive Bayes | 28.06% | 28.98% | 0.721 |
| Naive Bayes - SM | 55.6% | 29.75% | 0.708 |
| Bagging | 86.24% | 84.28% | 0.794 |
| Bagging - SM | 74.5% | 71.81% | 0.785 |
| AdaBoosting | 83.88% | 83.95% | 0.752 |
| AdaBoosting - SM | 67.85% | 68.56% | 0.747 |
| Gradient Boosting | 84.77% | 84.13% | 0.774 |
| Gradient Boosting - SM | 71.77% | 71.78% | 0.769 |
| SVM | 85.29% | 84.31% | 0.754 |
| SVM - CV | 86.33% | 84.31% | 0.751 |
| SVM - SM | 74.09% | 71.99% | 0.75 |

### 🏆 Best Model — KNN (Default, N_neighbours = 5)
**Conclusion from Final Report:** *"KNN with default values outperforms all other models built, based on Accuracy, F1 score, Recall, Precision and AUC score."*
- **Test Accuracy: 84.04%**
- **Train Accuracy: 85.72%**
- **AUC Score (Training): 0.749 | AUC Score (Testing): 0.715**
- 70:30 train-test stratified split; SMOTE applied only on training data
- Dataset: 7,882 train / 3,378 test (before SMOTE) | 13,112 train (after SMOTE)

## 💡 Business Impact & Recommendations
- **Maximum churn from "Regular+" account segment** — targeted offers needed
- **Single customers** contribute highest churn — bundle family plans recommended
- Customers in **Tier-1 cities** have highest computer usage; visibility and UX investment needed
- **Transactions via UPI and e-wallet are very low** — promote digital payment discounts
- Complaints in the last 12 months show no direct churn correlation — monitor service quality
- **Four Stages of Churn Management**: Acquire → Delight → Prevent → Retain
- Businesses can bifurcate customers by spending patterns (Deal seeker / Tariff optimizer) for targeted strategies

## 💡 Key Learnings
- Real-world data cleaning (special characters, mixed types, high-null columns)
- SMOTE for class imbalance (9,364 non-churners vs 1,896 churners)
- Stratified 70:30 split to preserve class distribution
- Multi-model comparison: default, GridSearchCV, SMOTE variants
- ROC-AUC curve plotting and interpretation
- Business translation of model outputs into Four-Stage Churn Management strategy

## 👤 Author
Mahesh G — Data Analyst & ML Engineer | MBA Data Science & Analytics, Jain University

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![ML](https://img.shields.io/badge/ML-KNN%20Best-green) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
