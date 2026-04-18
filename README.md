# Customer Churn Analysis Demo

Customer retention is a major and important business topic for many industries and businesses. Losing customers reduces long-term growth opportunities and can increase marketing costs. This project analyzes customer churn behavior, identifies key churn drivers, builds predictive models, and provides business recommendations. The dataset is the Kaggle Telco Customer Churn dataset.

## Project Goal

This repository provides a clean, reproducible customer churn analysis workflow. The project focuses on:

- understanding churn behavior in a telecom customer base
- cleaning the source dataset conservatively
- training baseline predictive models for churn
- generating business-friendly outputs and charts
- translating model findings into retention recommendations

## Dataset Source

- Dataset: Telco Customer Churn
- Source: Kaggle
- Common reference: IBM sample telecom churn dataset distributed through Kaggle
- Raw input file used by this project: `data/customer_churn.csv`

The dataset contains 7,043 customer records and 21 columns, including service attributes, contract details, billing behavior, and churn status.

## Data Cleaning Steps

The project uses conservative cleaning so the original business signal is preserved:

1. Standardize column names to `snake_case`.
2. Trim whitespace from text fields.
3. Convert null-like string values such as `na`, `null`, and `missing` to proper missing values.
4. Coerce known numeric fields such as `tenure`, `monthlycharges`, `totalcharges`, and `seniorcitizen` to numeric types.
5. Preserve limited missing values instead of dropping rows aggressively.
6. Keep raw data in `data/` unchanged and write processed outputs to `outputs/`.

One known issue in the source data is that `totalcharges` has 11 missing values for zero-tenure customers. Those values are retained and handled with model-time imputation.

## Modeling Approach

The modeling workflow is designed to stay simple, readable, and reproducible:

1. Load the cleaned churn dataset.
2. Encode the churn target as a binary outcome.
3. Exclude identifier leakage by removing `customerid` from modeling.
4. Split the data into stratified train and test sets.
5. Compare two baseline classifiers using 5-fold cross-validated ROC AUC:
   - logistic regression
   - random forest
6. Select the better-performing model and evaluate it on the held-out test set.
7. Save metrics, plots, and written summaries under `outputs/`.

Categorical variables are handled with one-hot encoding. Numeric features use median imputation, and the logistic regression pipeline also applies scaling.

## Key Results

The current pipeline selected `logistic_regression` as the best baseline model.

- Test ROC AUC: `0.841`
- Cross-validated ROC AUC: `0.846 +/- 0.012`
- Accuracy: `0.738`
- Precision: `0.504`
- Recall: `0.783`
- F1 score: `0.614`
- Overall churn rate: `26.5%`

Key observed churn patterns:

- Month-to-month customers have the highest churn rate at `42.7%`.
- One-year contract customers churn at `11.3%`.
- Two-year contract customers churn at `2.8%`.
- Fiber optic customers show the highest churn rate by internet segment at `41.9%`.

## Business Recommendations

- Prioritize retention offers for month-to-month customers, since they are materially more likely to churn than longer-contract customers.
- Build targeted conversion campaigns that encourage migration from month-to-month plans to one-year or two-year contracts.
- Review service quality, pricing, and support experience for fiber optic customers, who appear to be the highest-risk internet segment.
- Focus onboarding and early-life engagement on new customers, especially those with low tenure and flexible contracts.
- Use the churn model as a prioritization tool for outreach rather than as a fully automated decision system.

## Project Outputs

Running the pipeline generates the following artifacts in `outputs/`:

- `cleaned_data.csv`
- `metrics.json`
- `churn_distribution.png`
- `feature_importance.png`
- `model_performance.png`
- `analysis_summary.md`

## How To Run The Project

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the pipeline:

```bash
python -m app.run_pipeline
```

3. Review the generated files in `outputs/`.

## Repository Structure

```text
app/
  clean_data.py
  train_model.py
  make_charts.py
  summarize_results.py
  run_pipeline.py
data/
  customer_churn.csv
outputs/
docs/
```
