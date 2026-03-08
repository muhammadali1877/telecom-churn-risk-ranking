# Telecom Customer Churn Risk Ranking System

## Overview

Customer churn is a major problem in the telecom industry because losing existing customers is far more expensive than acquiring new ones. This project builds a machine learning pipeline to predict the probability that a customer will churn and ranks customers by risk level.

The system allows analysts to upload a customer dataset and automatically generate a churn risk ranking. Customers with the highest probability of churn can then be targeted with retention campaigns.

The model was built using a Logistic Regression pipeline with feature engineering, hyperparameter tuning, and threshold optimization.

---

## Key Features

* End-to-end machine learning pipeline using Scikit-learn
* Data preprocessing using `Pipeline` and `ColumnTransformer`
* Feature engineering (`Avg Monthly Spend`)
* Hyperparameter tuning using `GridSearchCV`
* Decision threshold optimization to improve churn recall
* Customer risk segmentation
* Streamlit web application for batch scoring
* Exportable churn risk ranking report

---

## Project Structure

```
telecom-churn-app/
│
├── app.py                         # Streamlit application
├── telecom_churn_model.pkl        # Trained model + threshold
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
```

---

## Machine Learning Workflow

The project follows a complete ML lifecycle:

```
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Preprocessing Pipeline
        ↓
Model Training
        ↓
Hyperparameter Tuning
        ↓
Threshold Optimization
        ↓
Final Model Training
        ↓
Customer Risk Ranking
        ↓
Streamlit Deployment
```

---

## Feature Engineering

A derived feature was created to better represent customer spending behavior:

```
Avg Monthly Spend = Total Charges / (Tenure Months + 1)
```

This feature captures customer spending intensity and improved model performance.

---

## Model

Algorithm used:

```
Logistic Regression
```

Best hyperparameters from tuning:

```
C = 10
class_weight = None
solver = liblinear
```

Evaluation results:

| Metric               | Value |
| -------------------- | ----- |
| Accuracy             | ~0.81 |
| ROC-AUC              | ~0.85 |
| Recall (Churn Class) | ~0.57 |

After threshold optimization:

```
Decision Threshold = 0.30
```

This increases churn detection significantly.

---

## Risk Segmentation

Customers are categorized based on predicted churn probability:

| Probability | Risk Level     |
| ----------- | -------------- |
| ≥ 0.70      | Very High Risk |
| ≥ 0.50      | High Risk      |
| ≥ 0.30      | Medium Risk    |
| < 0.30      | Low Risk       |

---

## Streamlit Application

The Streamlit dashboard allows analysts to:

1. Upload a CSV file containing customer data
2. Generate churn probabilities
3. Apply the optimized decision threshold
4. Rank customers by churn risk
5. Download a risk ranking report

---

## Running the App Locally

Install dependencies:

```
pip install -r requirements.txt
```

Run the Streamlit app:

```
streamlit run app.py
```

Then open the URL shown in the terminal.

---

## Input Dataset Requirements

The uploaded CSV should include the following columns:

```
Gender
Senior Citizen
Partner
Dependents
Tenure Months
Phone Service
Multiple Lines
Internet Service
Online Security
Online Backup
Device Protection
Tech Support
Streaming TV
Streaming Movies
Contract
Paperless Billing
Payment Method
Monthly Charges
Total Charges
CLTV
```

---

## Output

The application generates a ranked dataset with additional columns:

```
Churn Probability
Predicted Churn
Risk Segment
```

The results can be exported as a CSV file for further analysis.

---

## Business Value

This system enables telecom companies to:

* Identify customers likely to churn
* Prioritize retention efforts
* Reduce customer loss
* Improve marketing efficiency

Instead of reacting to churn, the company can take proactive action.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Joblib

---

## Future Improvements

Potential enhancements include:

* Random Forest or Gradient Boosting models
* SHAP feature importance analysis
* Automated data validation
* Real-time API scoring
* Customer retention recommendation system

---

## Author

Muhammad Ali Azhar
AI / Machine Learning Enthusiast
