# 📊 Customer Churn Prediction – Machine Learning Project

## 📌 Project Overview

Customer churn is a major challenge in the telecom industry, where retaining existing customers is more cost-effective than acquiring new ones.
This project builds a **machine learning–based classification system** to predict whether a customer is likely to churn based on their demographic details, account information, and service usage.

The project demonstrates a **complete end-to-end ML workflow**, from data preprocessing to model evaluation and insights.

---

## 🎯 Objective

* Predict whether a customer will **churn (1)** or **not churn (0)**
* Identify the **most reliable classification model**
* Gain actionable insights to support customer retention strategies

---

## 🗂 Dataset

* **Source:** Telco Customer Churn Dataset (Kaggle)
* **Records:** ~7,000 customers
* **Features:**

  * Demographics (gender, senior citizen)
  * Account details (tenure, contract type, payment method)
  * Services subscribed (internet, phone, streaming)
* **Target Variable:** `Churn` (Binary)

---

## 🛠 Tech Stack

* **Language:** Python
* **Libraries:**

  * `pandas`, `numpy` – Data manipulation
  * `matplotlib`, `seaborn` – Visualization
  * `scikit-learn` – Modeling & evaluation

---

## 🔄 Project Workflow

1. Data loading and inspection
2. Data cleaning and preprocessing
3. Feature encoding and scaling
4. Train–test split
5. Model training
6. Model evaluation
7. Cross-validation
8. Final insights and conclusions

---

## 🧹 Data Preprocessing

### Key Steps:

* Dropped `customerID` (non-predictive identifier)
* Converted `TotalCharges` from string to numeric
* Handled missing values using **median imputation**
* Applied **One-Hot Encoding** for categorical features
* Used **StandardScaler** for feature scaling
* Stratified train–test split (80/20) to preserve churn distribution

---

## 🤖 Models Implemented

The following models were trained and evaluated:

### 1️⃣ Logistic Regression

* Baseline linear classifier
* Interpretable and efficient
* Provides churn probability estimates
* Used with increased `max_iter` for convergence

### 2️⃣ Decision Tree Classifier

* Non-linear, rule-based model
* Captures feature interactions
* Depth controlled to reduce overfitting

### 3️⃣ Random Forest Classifier

* Ensemble of decision trees
* Reduces variance compared to a single tree
* Handles non-linear patterns effectively

---

## 📈 Model Evaluation Metrics

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

Additionally:

* **5-Fold Cross-Validation** applied to Logistic Regression using **F1-Score**

---

## 📊 Model Performance Comparison

| Model               | Key Observations                                                    |
| ------------------- | ------------------------------------------------------------------- |
| Logistic Regression | Most stable and consistent performance                              |
| Decision Tree       | Slight overfitting despite depth control                            |
| Random Forest       | Competitive but no significant improvement over Logistic Regression |

---

## ✅ Final Conclusion

**Logistic Regression** emerged as the **most reliable model** for this project due to:

* Consistent F1-score across cross-validation
* Good balance between precision and recall
* Lower risk of overfitting
* High interpretability for business decision-making

Although Random Forest performed well, it **did not significantly outperform Logistic Regression** in this implementation.

---

## 💡 Key Insights

### Technical Insights

* Proper preprocessing has a major impact on model performance
* Simpler models can outperform complex models on well-structured data
* Cross-validation is essential for reliable model selection

### Business Insights

* Customers with **short tenure** are more likely to churn
* **Month-to-month contracts** show higher churn rates
* Contract type and payment method strongly influence customer retention


