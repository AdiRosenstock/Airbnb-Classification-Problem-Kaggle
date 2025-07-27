# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from catboost import CatBoostClassifier

# Reading the data
train = pd.read_csv("train_clas.csv")
test = pd.read_csv("test_clas.csv")

# Target variable
y_train = train["host_is_superhost"]

# Dropping features
X_train = train.drop(columns=["id", "host_is_superhost","description","host_about"])
X_test = test.drop(columns=["id","description","host_about"])

# Cleaning the data
for col in ['host_response_rate', 'host_acceptance_rate']:
    X_train[col] = X_train[col].str.rstrip('%').astype(float)
    X_test[col] = X_test[col].str.rstrip('%').astype(float)
    
def extract_bathroom_count(x):
    try:
        return float(x.split(' ')[0])
    except:
        return None

X_train['bathrooms_text'] = X_train['bathrooms_text'].apply(extract_bathroom_count)
X_test['bathrooms_text'] = X_test['bathrooms_text'].apply(extract_bathroom_count)


# Filling missing data

# Numerical features
num_cols = X_train.select_dtypes(include='number').columns

for col in num_cols:
    X_train[col] = X_train[col].fillna(X_train[col].mean())
    X_test[col] = X_test[col].fillna(X_train[col].mean())

# Categorical features
cat_features = X_train.select_dtypes(exclude='number').columns.tolist()

for col in cat_features:
    X_train[col] = X_train[col].astype(str).fillna("missing")
    X_test[col] = X_test[col].astype(str).fillna("missing")

# used for fitting the model
cat_feature_indices = [X_train.columns.get_loc(col) for col in cat_features]


# Training and fitting the model
model = CatBoostClassifier(
    random_state = 1,
    n_estimators = 500, 
    learning_rate = 0.05,
    max_depth = 8,
    subsample = 0.75,
    reg_lambda = 1,
    verbose = False,
    scale_pos_weight= 5,
    thread_count = 1
)

model.fit(X_train, y_train, cat_features=cat_feature_indices)
test_probs = model.predict_proba(X_test)[:, 1]

# Creating the submission file
submission_classification = pd.DataFrame({'id': test['id'],'predicted': test_probs})
submission_classification.to_csv('submission_classification.csv', index=False)