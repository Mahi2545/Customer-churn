# Customer Churn Prediction

## 📊 Overview
Classification model to predict customer churn using machine learning. This project analyzes customer behavior patterns to identify at-risk customers and inform retention strategies.

## 🎯 Objective
Build a predictive model to identify customers likely to churn, enabling proactive retention efforts and business strategy optimization.

## 🛠️ Technologies Used
- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Scikit-learn** - ML algorithms
- **Matplotlib/Seaborn** - Visualizations
- **Jupyter Notebook** - Analysis environment

## 📁 Project Structure
Customer-churn/
├── data/
│ ├── churn.csv
│ └── data_description.txt
├── notebooks/
│ ├── exploratory_analysis.ipynb
│ └── model_training.ipynb
├── src/
│ ├── preprocessing.py
│ ├── feature_engineering.py
│ └── model.py
└── README.md


## 🔍 Key Features
- **EDA**: Comprehensive exploratory data analysis
- **Class Imbalance Handling**: SMOTE or class weights
- **Feature Selection**: Identifying important predictors
- **Multiple Models**: Logistic Regression, Random Forest, Gradient Boosting
- **ROC-AUC Analysis**: Model performance evaluation
- **Business Metrics**: Precision, Recall, F1-Score

## 📊 Target Variable
**Churn**: Whether customer discontinued service (Yes/No)

## 🔍 Key Predictors
- Tenure
- Monthly charges
- Contract type
- Internet service type
- Customer support usage
- Additional services

## 📈 Models Evaluated
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting
- XGBoost
- SVM

## 🚀 Usage
Install dependencies
pip install -r requirements.txt

Run analysis
jupyter notebook notebooks/exploratory_analysis.ipynb

Train model
python src/model.py


## 📊 Model Performance
- Accuracy: [Your Score]
- Precision: [Your Score]
- Recall: [Your Score]
- F1-Score: [Your Score]
- ROC-AUC: [Your Score]

## 💡 Business Impact
✅ Identifies high-risk churn customers
✅ Enables targeted retention campaigns
✅ Improves customer lifetime value
✅ Optimizes marketing spend
✅ Reduces revenue loss

## 🎯 Recommendations
1. Focus retention efforts on high-churn-risk segments
2. Improve customer support for at-risk groups
3. Develop loyalty programs
4. Monitor model performance quarterly

## 👤 Author
Mahesh G - Data Analyst & ML Engineer

---

*Classification Problem - Customer Retention Analytics*
