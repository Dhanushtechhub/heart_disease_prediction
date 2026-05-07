# ❤️ Heart Disease Prediction using Machine Learning

A classification project that predicts heart disease using four ML algorithms, with full EDA, hyperparameter tuning, and feature importance analysis.

## 📁 Project Structure

heart_disease_prediction/
├── heart_disease_prediction.ipynb   # Main notebook
├── heart_disease_dataset.csv        # Dataset (200 rows, 13 features)
├── requirements.txt
└── README.md

## 📊 Dataset

| Property | Value |
|---|---|
| Rows | 200 |
| Features | 13 |
| Target | `target` (0 = No Disease, 1 = Disease) |

Features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal

## 🔬 Workflow

1. EDA — target distribution, correlation heatmap
2. Preprocessing — StandardScaler, 80/20 train-test split
3. Model Training — 4 classifiers trained and evaluated
4. Cross-Validation — 5-fold CV for unbiased accuracy estimates
5. Hyperparameter Tuning — GridSearchCV on Random Forest
6. Feature Importance — Random Forest vs XGBoost comparison

## 🤖 Models & Results

| Model | CV Accuracy |
|---|---|
| Logistic Regression | ~0.53 |
| Random Forest | ~0.58 |
| SVM | ~0.51 |
| XGBoost | ~0.56 |

> Note: The dataset has only 200 rows with weak feature-target correlations (max 0.12),
> which limits performance across all models. This is a dataset limitation, not a modeling issue.

## 🛠️ Requirements

- Python 3.8+
- scikit-learn
- xgboost
- pandas, numpy
- matplotlib, seaborn
- jupyter
