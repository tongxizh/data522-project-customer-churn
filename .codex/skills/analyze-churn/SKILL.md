---
name: analyze-churn
description: Clean a customer churn dataset, train a baseline model, generate plots, and summarize results.
---

# When to use
Use this skill when working with customer churn CSV data.

# Workflow
1. Load the dataset from `data/customer_churn.csv`.
2. Clean the dataset conservatively.
3. Ensure the churn target is binary.
4. Train a baseline classification model.
5. Save metrics to `outputs/metrics.json`.
6. Generate plots to `outputs/`.
7. Write a concise markdown summary.

# Guardrails
- Do not overwrite raw data.
- Do not drop columns aggressively.
- Do not do heavy feature engineering in v1.
- Prefer reproducibility over cleverness.

# Deliverables
- outputs/cleaned_data.csv
- outputs/metrics.json
- outputs/churn_distribution.png
- outputs/feature_importance.png
- outputs/model_performance.png
- outputs/analysis_summary.md