# importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Load Train and Test Datasets 
train = pd.read_csv('bank-full.csv', sep=";")
test = pd.read_csv('bank.csv', sep=";")

print(train.shape, test.shape) 
print(train.head())

# Data Cleaning 
# Drop duplicates 
train = train.drop_duplicates() 
test = test.drop_duplicates()

# Normalise Train and Test  Numerical Attributes

from sklearn.preprocessing import MinMaxScaler

attributes_to_normalize = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
data_to_normalize = train[attributes_to_normalize]
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data_to_normalize)
train[attributes_to_normalize] = normalized_data

train[['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']].describe()

attributes_to_normalize = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
data_to_normalize = test[attributes_to_normalize]
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data_to_normalize)
test[attributes_to_normalize] = normalized_data

test[['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']].describe() 


# Encode target variable
train["y"] = train["y"].map({"yes": 1, "no": 0}) 
test["y"] = test["y"].map({"yes": 1, "no": 0})

#  Feature Engineering 
for df in [train, test]: 
    df["age_group"] = pd.cut( 
        df["age"], 
        bins=[0, 24, 34, 44, 54, 64, 100], 
        labels=["Young", "Young Adult", "Adult", "Middle-aged", "Senior", "Retired"] 
    )


# Convert month to ordinal for both train and test
month_order = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
               'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
train['month'] = train['month'].map(month_order)

month_order = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
               'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
test['month'] = test['month'].map(month_order)

# Define X_train and y_train
X_train = train.drop(columns='y', axis=1)
y_train = train['y']

# Define X_test and y_test
X_test = test.drop(columns='y', axis=1)
y_test = test['y']

print('Training Data Set Shape:', X_train.shape, y_train.shape)

# Identify columns
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessing
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

from imblearn.over_sampling import ADASYN

# Apply preprocessing to X_train
X_train_processed = preprocessor.fit_transform(X_train)

# Resample with ADASYN
adasyn = ADASYN(random_state=42, sampling_strategy=0.8)
X_resampled, y_resampled = adasyn.fit_resample(X_train_processed, y_train)

print('Resampled Data Set Shape:', X_resampled.shape, y_resampled.shape)
print(y_resampled.value_counts())

# Get new feature names from preprocessor
feature_names = preprocessor.get_feature_names_out()

# Create DataFrame with resampled features + target
X_df_resampled = pd.DataFrame(X_resampled, columns=feature_names)
X_df_resampled['y'] = y_resampled

print(X_df_resampled.head())


from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import ADASYN

# --- Logistic Regression Pipeline with ADASYN ---
log_reg_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('adasyn', ADASYN(random_state=42, sampling_strategy=0.8)),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])

# Fit on raw training data
log_reg_pipeline.fit(X_train, y_train)

# Predict on test set
y_pred_log = log_reg_pipeline.predict(X_test)
y_prob_log = log_reg_pipeline.predict_proba(X_test)[:, 1]

# Evaluate
print("\n=== Logistic Regression with ADASYN ===")
print(classification_report(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("ROC AUC:", roc_auc_score(y_test, y_prob_log))


# --- Random Forest Pipeline with ADASYN ---
rf_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('adasyn', ADASYN(random_state=42, sampling_strategy=0.8)),
    ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
])

# Fit on raw training data
rf_pipeline.fit(X_train, y_train)

# Predict on test set
y_pred_rf = rf_pipeline.predict(X_test)
y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]

# Evaluate
print("\n=== Random Forest with ADASYN ===")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("ROC AUC:", roc_auc_score(y_test, y_prob_rf))



# Save the Random Forest pipeline
joblib.dump(rf_pipeline, 'rf_bank_model.pkl')













