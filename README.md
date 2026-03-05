# Customer Churn Prediction

## 📊 Overview
End-to-end classification project to predict customer churn using machine learning on a real-world DTH customer dataset (11,000+ records, 19 features). The project identifies at-risk customers and provides actionable retention strategies.

## 🎯 Objective
Build a predictive model to identify customers likely to churn, enabling proactive retention efforts and business strategy optimization.

## 🛠️ Technologies Used
- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Scikit-learn** - ML algorithms (LR, RF, SVC, KNN)
- **XGBoost** - Best performing classifier
- **Matplotlib/Seaborn** - Visualizations
- **Jupyter Notebook** - Analysis environment

## 🔍 Key Features
- **EDA**: Comprehensive analysis of churn drivers — city tier, tenure, support interactions, payment preferences, account segment
- **Data Cleaning**: Special character treatment, missing value imputation (dropped cols >30% missing), category standardization
- **Encoding**: One-Hot Encoding for categorical variables (drop_first=True), StandardScaler for feature normalization
- **Class Imbalance**: Handled via stratified train-test split
- **Multiple Models**: 5 classifiers evaluated — Logistic Regression, Random Forest, SVC, KNN, XGBoost
- **ROC-AUC Analysis**: Full ROC curve plotted for best model
- **Business Segmentation**: At-risk customers segmented by city tier and revenue potential

## 📈 Key Predictors
- Tenure
- Monthly charges / Revenue
- Account segment (Regular, Regular Plus, Super, Super Plus)
- City tier
- Customer support call frequency
- Payment method
- Marital status
- Gender

## 📈 Model Performance Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.78 | ~0.76 | ~0.72 | ~0.74 | ~0.85 |
| K-Nearest Neighbors | ~0.80 | ~0.78 | ~0.74 | ~0.76 | ~0.87 |
| Support Vector Classifier | ~0.81 | ~0.79 | ~0.75 | ~0.77 | ~0.88 |
| Random Forest Classifier | ~0.84 | ~0.82 | ~0.79 | ~0.80 | ~0.91 |
| **XGBoost Classifier (Best)** | **~0.86** | **~0.84** | **~0.81** | **~0.82** | **~0.92** |

### 🏆 Best Model — XGBoost Classifier
- **Accuracy: ~86%**
- **ROC-AUC: ~0.92**
- **Precision: ~84%** | **Recall: ~81%** | **F1-Score: ~82%**
- ROC Curve plotted and validated on held-out test set (80/20 stratified split)

## 💡 Business Impact
- Identified high-risk churn customer segments by city tier and account type
- Proposed targeted retention campaigns for Super Plus and Regular Plus segments
- Recommended loyalty programs for high-tenure, high-revenue customers
- Enabled data-driven prioritization of customer support interventions
- Potential to reduce revenue loss by focusing retention spend on predicted churners

## 🎯 Recommendations
1. Focus retention efforts on high-churn-risk city-tier 1 customers with short tenure
2. Improve customer support response time for at-risk account segments
3. Develop loyalty programs targeting Super and Super Plus customers
4. Monitor model performance quarterly and retrain on new data

## 💡 Key Learnings
- Real-world data cleaning (special chars, mixed types, high-null columns)
- One-Hot Encoding and StandardScaler pipeline
- Stratified splitting for class balance preservation
- Multi-model comparison with Accuracy, Precision, Recall, F1, ROC-AUC
- Business translation of model outputs into retention strategy

## 👤 Author
Mahesh G — Data Analyst & ML Engineer

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![ML](https://img.shields.io/badge/ML-XGBoost-green) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
