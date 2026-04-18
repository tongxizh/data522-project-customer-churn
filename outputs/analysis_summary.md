# Customer Churn Analyst Report

## Executive Summary
- The dataset contains 7,043 customers and 21 columns after conservative cleaning.
- Overall churn is 26.5%, which creates a moderate class imbalance but remains suitable for a baseline classifier.
- The selected model was `logistic_regression`, with test AUC of 0.841 and cross-validated AUC of 0.846 +/- 0.012.

## Data Quality
- Remaining missingness is limited and was preserved conservatively rather than dropped:
- `totalcharges`: 11 rows (0.2%)
- Identifier fields excluded from modeling: customerid.

## Model Readout
- Accuracy: 0.738
- Precision: 0.504
- Recall: 0.783
- F1 score: 0.614

## Business Signals
- Month-to-month customers have the highest churn rate at 42.7%, well above longer-term contracts.
- Customers with `Fiber optic` internet show the highest churn rate at 41.9%.
- The first retention levers to investigate are contract conversion, onboarding for new accounts, and service experience in the highest-risk product segments.