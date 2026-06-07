# XGBoost_demo

Main notebook: `telco_customer_churn_analysis.ipynb`

## Tools used
- **Python / Jupyter Notebook** for end-to-end analysis workflow
- **pandas, numpy** for data loading and feature preparation
- **scikit-learn** for preprocessing, pipelines, model training, and evaluation
- **xgboost** for gradient-boosted tree classification
- **shap** for model explainability
- **matplotlib, seaborn** for data and metric visualization

## Main concepts implemented
- **Data cleaning**: converts `TotalCharges` to numeric and handles missing/invalid values
- **Target encoding**: maps `Churn` (`Yes`/`No`) to binary labels
- **Feature engineering**: behavior/payment proxy features (for example, `AvgMonthlyCharge`, fiber/e-check indicators)
- **Preprocessing pipeline**:
  - Median imputation + scaling for numeric features
  - Most-frequent imputation + one-hot encoding for categorical features
  - Unified with `ColumnTransformer` and `Pipeline`
- **Model comparison**: Logistic Regression, Decision Tree, Random Forest, and XGBoost
- **Evaluation metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Model selection**: best model chosen by ROC-AUC
- **Explainability**: SHAP summary for the selected best model
