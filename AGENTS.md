# AGENTS.md

## Project purpose
This repository is for customer churn analysis with an AI-assisted workflow.

## Working rules
- Never overwrite raw files in `data/`.
- Save all artifacts to `outputs/`.
- Prefer small, readable Python scripts.
- Keep the first version simple and reproducible.
- If assumptions are needed, state them clearly.

## Standard workflow
1. Load the churn CSV from `data/`.
2. Clean obvious data issues conservatively.
3. Train a baseline churn model.
4. Generate clear plots.
5. Write a short markdown report.

## Deliverables
- outputs/cleaned_data.csv
- outputs/metrics.json
- outputs/churn_distribution.png
- outputs/feature_importance.png
- outputs/model_performance.png
- outputs/analysis_summary.md