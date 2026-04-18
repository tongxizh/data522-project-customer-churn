from app.clean_data import clean_dataset
from app.make_charts import (
    plot_feature_importance,
    plot_model_performance,
    plot_target_distribution,
)
from app.summarize_results import summarize
from app.train_model import train_churn_model

INPUT_PATH = "data/customer_churn.csv"
CLEANED_PATH = "outputs/cleaned_data.csv"
METRICS_PATH = "outputs/metrics.json"
SUMMARY_PATH = "outputs/analysis_summary.md"

TARGET_COL = "churn"


def main():
    df = clean_dataset(INPUT_PATH, CLEANED_PATH)
    pipeline, X_test, y_test, preds, probs, _ = train_churn_model(df, TARGET_COL, METRICS_PATH)
    plot_target_distribution(df, TARGET_COL, "outputs/churn_distribution.png")
    plot_model_performance(y_test, probs, preds, "outputs/model_performance.png")
    plot_feature_importance(pipeline, "outputs/feature_importance.png")
    summarize(df, TARGET_COL, METRICS_PATH, SUMMARY_PATH)


if __name__ == "__main__":
    main()
