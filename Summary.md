# 🏡 Airbnb Superhost Classification | STAT 303-3 ML Competition

This repository contains my submission for the final machine learning competition in **STAT 303-3: Data Science with Python III** at Northwestern University. The objective was to predict whether an Airbnb host is a **Superhost** using real listing data from Chicago, Asheville, and Kauai Island.

> 🥈 **Final Result**: 2nd out of 124 participants  
> 📈 **ROC-AUC Score**: 0.9908

---

## 📌 Problem Statement

Each Airbnb listing is described by 56–57 features (e.g., pricing, reviews, availability, amenities). The task is to build a classification model that estimates the **probability** that a host is a Superhost (`host_is_superhost` = 1).

- **Goal**: Binary classification with probabilistic output
- **Evaluation Metric**: ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
- **Submission Format**: Probability predictions for test data

---

## 🧾 Data Description

The dataset consists of anonymized **Airbnb listings** from three markets: **Chicago, IL**, **Asheville, NC**, and **Kauai Island, HI**.


### 🔹 Training Set:
- **Rows**: 5,510 listings
- **Columns**: 57 (including the target)
  - `host_is_superhost`: Binary target variable (1 = superhost, 0 = not)
  - `id`: Unique listing identifier
  - 55+ other features including numeric, categorical, text, and boolean variables

### 🔹 Test Set:
- **Rows**: 3,308 listings
- **Columns**: 56 (same features, but without `host_is_superhost`)

### 📌 Notable Feature Types:
- **Categorical**: `property_type`, `room_type`, `neighbourhood_cleansed`, etc.
- **Numeric**: `price`, `number_of_reviews`, `availability_365`, etc.
- **Text/Complex**: `amenities`, `description` (used for feature engineering)
- **Boolean/Flags**: `host_identity_verified`, `instant_bookable`, etc.

---

## 🧹 Data Preprocessing

- Cleaned inconsistent formats (e.g., `$`, `%`, string categories)
- Imputed missing values (median/mode or flagged)
- Encoded categorical variables (One-Hot, Frequency Encoding)
- Scaled numerical features
- Engineered custom features based on host behavior and availability

---

## 🤖 Modeling Approach

All models were tuned using **Stratified K-Fold Cross-Validation (5 folds)** to ensure class balance and reliable out-of-sample estimates. The final predictions were submitted using the tuned **CatBoostClassifier**, which performed best on the validation set.


### Final Model: CatBoostClassifier

```python
model = CatBoostClassifier(
    random_state=1,
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.75,
    reg_lambda=1,
    verbose=False,
    scale_pos_weight=5,
    thread_count=1
)
```

In addition to the final CatBoostClassifier, I explored and tuned several other models to compare performance. These included Random Forest, Bagged Trees, Decision Trees, Adapative Boosting and XGBoost. I also experimented with a stacked ensemble that combined top performing models. While each model had its strengths, CatBoost consistently outperformed all others in ROC-AUC, especially in handling categorical features and imbalanced data.


---

## 📈 Performance Summary

| Rank | Score (ROC-AUC) | Participants |
|------|------------------|--------------|
| 🥈 **2nd Place** | **0.9908** | 124 students |

---

## ⚠️ Disclaimers

- The full code and modeling process is documented in the accompanying `.html` notebook (`Airbnb_Classification_Problem.html`) located in the repository.
- Due to computational demands, model tuning (especially for ensemble methods like XGBoost and CatBoost) was performed using a remote console environment with higher CPU and memory capacity. This ensured faster grid searches, better resource handling, and overall smoother experimentation.
- The data set was provided by kaggle
