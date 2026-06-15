# Credit Risk ML Classifier

> Binary classification pipeline for loan approval decisions  
> Python · scikit-learn · GridSearchCV · Leakage-safe Pipeline · ROC-AUC 0.97

## Overview

Comparative study of three classification models on a structured credit dataset of 45,000 loan applications. The task is to predict loan approval outcomes (`loan_status`) from applicant demographics, financial capacity, and credit history.

Built for UCL MSc Financial Technology — COMP0198 Machine Learning with Applications in Finance.

## Models Compared

| Model | ROC-AUC |
|---|---|
| Dummy Classifier (baseline) | 0.500 |
| Logistic Regression | 0.950 |
| Artificial Neural Network | 0.955 |
| **Random Forest** ✅ | **0.971** |

Random Forest is selected as the final model based on highest ROC-AUC, robustness to class imbalance, and stability across cross-validation folds.

## Key Design Decisions

- **Leakage-safe preprocessing**: all scaling and encoding fitted exclusively on training data via `sklearn.Pipeline`
- **Stratified train/test split**: preserves 22.2% minority class ratio across subsets
- **ROC-AUC as primary metric**: chosen over accuracy due to pronounced class imbalance
- **Separate preprocessors**: Random Forest uses passthrough for numerical features (no scaling needed); Logistic Regression and ANN use `StandardScaler`
- **GridSearchCV tuning**: 5-fold cross-validation on held-out training data only; test set used once for final evaluation

## Results

**Random Forest** (best params: `max_depth=20`, `max_features='sqrt'`, `min_samples_leaf=1`):
- ROC-AUC: **0.9713**
- Top features: loan-to-income ratio, interest rate, prior default history, personal income

**Logistic Regression** (best params: `C=10`):
- ROC-AUC: 0.9503

**ANN** (architecture: 64→32, ReLU, Adam, L2 regularisation):
- ROC-AUC: 0.9552

## How to Run

```bash
pip install pandas numpy matplotlib scikit-learn
jupyter notebook COMP0198_Coursework.ipynb
```

> **Note**: The dataset (`Dataset_2526.csv`) was provided by UCL and is not included in this repository. The notebook expects it in the working directory. The dataset contains 45,000 observations across 16 features including applicant age, income, employment history, loan amount, interest rate, credit score, and prior default status.

## File Structure

```
├── COMP0198_Coursework.ipynb   # Full ML pipeline with outputs
└── README.md
```

## Deployment Considerations

- Decision thresholds should reflect asymmetric misclassification costs: false negatives (approving high-risk applicants) carry higher financial cost than false positives
- Features such as `person_age` and `person_gender` may act as proxies for protected characteristics — fairness review required before deployment
- Financial regulators require model explainability for adverse credit decisions; feature importance outputs from Random Forest support this requirement
