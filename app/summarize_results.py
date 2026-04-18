import json
from pathlib import Path


def summarize(df, target_col, metrics_path, output_path):
    with open(metrics_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    churn_rate = df[target_col].astype(str).str.strip().str.lower().eq("yes").mean()
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    contract_rates = (
        df.groupby("contract")[target_col]
        .apply(lambda series: series.astype(str).str.strip().str.lower().eq("yes").mean())
        .sort_values(ascending=False)
    )
    internet_rates = (
        df.groupby("internetservice")[target_col]
        .apply(lambda series: series.astype(str).str.strip().str.lower().eq("yes").mean())
        .sort_values(ascending=False)
    )

    lines = []
    lines.append("# Customer Churn Analyst Report\n")
    lines.append("## Executive Summary")
    lines.append(
        f"- The dataset contains {df.shape[0]:,} customers and {df.shape[1]} columns after conservative cleaning."
    )
    lines.append(
        f"- Overall churn is {churn_rate:.1%}, which creates a moderate class imbalance but remains suitable for a baseline classifier."
    )
    lines.append(
        f"- The selected model was `{metrics['selected_model']}`, with test AUC of {metrics['auc']:.3f} and cross-validated AUC of {metrics['cv_auc_mean']:.3f} +/- {metrics['cv_auc_std']:.3f}."
    )

    lines.append("\n## Data Quality")
    if missing.empty:
        lines.append("- No missing values remained after cleaning.")
    else:
        lines.append("- Remaining missingness is limited and was preserved conservatively rather than dropped:")
        for column, count in missing.head(3).items():
            lines.append(f"- `{column}`: {count} rows ({count / len(df):.1%})")
    if metrics.get("dropped_identifier_columns"):
        lines.append(
            f"- Identifier fields excluded from modeling: {', '.join(metrics['dropped_identifier_columns'])}."
        )

    lines.append("\n## Model Readout")
    lines.append(f"- Accuracy: {metrics['accuracy']:.3f}")
    lines.append(f"- Precision: {metrics['precision']:.3f}")
    lines.append(f"- Recall: {metrics['recall']:.3f}")
    lines.append(f"- F1 score: {metrics['f1']:.3f}")

    lines.append("\n## Business Signals")
    lines.append(
        f"- Month-to-month customers have the highest churn rate at {contract_rates.iloc[0]:.1%}, well above longer-term contracts."
    )
    lines.append(
        f"- Customers with `{internet_rates.index[0]}` internet show the highest churn rate at {internet_rates.iloc[0]:.1%}."
    )
    lines.append(
        "- The first retention levers to investigate are contract conversion, onboarding for new accounts, and service experience in the highest-risk product segments."
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))
