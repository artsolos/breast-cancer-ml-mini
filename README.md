# Project: Breast Cancer Wisconsin ML Mini

## Goal
Predict benign vs malignant tumors from 30 numeric features. Deliver a trained model, test metrics, and a short explanation of drivers.

## Dataset
Breast Cancer Wisconsin (569 rows, 30 features).  
Target: 0 = malignant, 1 = benign.  
Source: built into scikit-learn.

## Method
- Stratified 80/20 split, seed 42.  
- Pipelines:  
  1. Logistic Regression (median imputation + scaling)  
  2. Random Forest (300 trees)  
- Evaluation: held-out test set + 5-fold cross-validation.  

## Results
**Logistic Regression**  
- Accuracy: 0.982  
- F1: 0.986  
- ROC-AUC: 0.995  
- CV ROC-AUC mean: 0.994, std: 0.011  

**Random Forest**  
- Accuracy: 0.947  
- F1: 0.958  
- ROC-AUC: 0.994  

## Key Drivers
- worst smoothness  
- worst area  
- worst concave points  
- area error  
- mean area  

## Files
- `breast_cancer_classification.ipynb`  
- `breast_cancer_lr.joblib`  

## How to Run in Colab
1. Open Colab.  
2. Upload the notebook.  
3. Run cells top to bottom.  
4. For a single prediction, run the last demo cell.  

## Limits
- Small, clean dataset.  
- No external validation.  
- Next steps: calibration curve, error analysis, XGBoost baseline.  
