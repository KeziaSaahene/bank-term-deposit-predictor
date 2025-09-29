# 📊 Bank Term Deposit Subscription Prediction

## 📌 Overview

This project predicts whether a client will subscribe to a **term deposit** (yes/no) using the **Bank Marketing dataset** from the UCI Machine Learning Repository.  


---

## 📂 Dataset

- **Source:** [Bank Marketing Dataset (UCI ML Repository)](https://archive.ics.uci.edu/ml/datasets/bank+marketing)  
- **Files:** `bank-full.csv` (train), `bank.csv` (test)  
- **Shape:** 41,188 rows × 21 columns (train), 4,521 rows × 21 columns (test)  
- **Target Variable:** `y` – yes (subscribed) or no (not subscribed)  

---

## ⚙️ Tech Stack

**Languages & Libraries:**  

- Python: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`  
- Streamlit for interactive web app  

**Models Used:**  

- Logistic Regression (with ADASYN balancing)  
- Random Forest Classifier (with ADASYN balancing)  

---

## 📊 Data Preprocessing & Feature Engineering

- **Duplicate removal**  
- **Normalization** for numerical columns: `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`  
- **Target encoding:** `yes → 1`, `no → 0`  
- **Age groups:** Young, Young Adult, Adult, Middle-aged, Senior, Retired  
- **Month conversion:** Jan → 1, Feb → 2, … Dec → 12  
- **Imbalanced data handling:** ADASYN oversampling (sampling strategy = 0.8)  
- **Pipeline creation:** `StandardScaler` + `OneHotEncoder` + ADASYN + Classifier  

---
- Preprocessing pipelines for both **numerical and categorical features**  
- **Feature engineering** (age grouping, month conversion to numeric)  
- Handling **imbalanced data** using **ADASYN**  
- Model training with **Logistic Regression** and **Random Forest**  
- Deployment via a **Streamlit app** with **light/dark mode**, **gradient probability bar**, and interactive numeric inputs  


## 🧑‍💻 Model Training & Evaluation

### Logistic Regression + ADASYN

- **Accuracy:** ~85%  
- **ROC AUC:** ~0.93  
- **Strengths:** High recall for subscribed clients  

### Random Forest + ADASYN

- **Accuracy:** ~91%  
- **ROC AUC:** ~0.94  
- **Strengths:** Better precision-recall trade-off compared to Logistic Regression  

---

## 🚀 Streamlit App Deployment

The app allows **interactive prediction**:  

- **Categorical inputs:** job, marital status, education, credit default, housing loan, personal loan, contact type, month, previous outcome  
- **Numeric inputs:** age, balance, day, duration, campaign, pdays, previous contacts  
- **Age group** and **month numeric conversion** handled automatically  
- **Gradient probability bar** shows likelihood of subscription  
- **Light/Dark theme toggle**  

**Try the app here:**  
[[https://bank-term-deposit-predictor-hkfnpcbe5bkrfnqqyhvhtj.streamlit.app](https://bank-term-deposit-predictor-hkfnpcbe5bkrfnqqyhvhtj.streamlit.app) ](https://bank-term-deposit-predictor-hkfnpcbe5bkrfnqqyhvhtj.streamlit.app) 

---
