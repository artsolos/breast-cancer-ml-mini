# Breast Cancer ML Mini Project

## Goal
Predict if a tumor is benign or malignant using a built-in dataset from scikit-learn. I trained a couple of models and compared the results.

## Dataset
- Breast Cancer Wisconsin dataset (569 samples, 30 features)
- Target: 0 = malignant, 1 = benign
- Already included in scikit-learn

## Method
- Train/test split 80/20 (stratified, random_state=42)
- Models:
  - Logistic Regression (with scaling + imputation)
  - Random Forest (300 trees)
- Metrics: Accuracy, F1, ROC-AUC + 5-fold CV

## Results
Logistic Regression
- Accuracy: 0.982
- F1: 0.986
- ROC-AUC: 0.995
- CV ROC-AUC ~0.994 (std ~0.01)

Random Forest
- Accuracy: 0.947
- F1: 0.958
- ROC-AUC: 0.994

## Important Features
- worst smoothness
- worst area
- worst concave points
- area error
- mean area

## Files
- breast_cancer_classification.ipynb
- breast_cancer_lr.joblib

## How to Run
1. Open notebook in Google Colab
2. Run all cells top to bottom
3. Last cell shows single prediction demo

## Notes
- Dataset is clean and small, so real-world performance may differ
- No external validation done
- Could try more models (like XGBoost) in future
