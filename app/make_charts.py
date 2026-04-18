from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

PLOT_STYLE = "whitegrid"


def _format_feature_name(name: str) -> str:
    cleaned = name.replace("num__", "").replace("cat__", "")
    return cleaned.replace("_", " ")


def plot_target_distribution(df, target_col, output_path):
    sns.set_theme(style=PLOT_STYLE)
    plot_df = df.copy()
    plot_df[target_col] = plot_df[target_col].astype(str).str.title()

    plt.figure(figsize=(7.5, 5.5))
    ax = sns.countplot(
        x=target_col,
        data=plot_df,
        order=sorted(plot_df[target_col].dropna().unique()),
        hue=target_col,
        palette="Blues",
        legend=False,
    )
    total = len(plot_df)

    for patch in ax.patches:
        count = int(patch.get_height())
        share = count / total
        ax.annotate(
            f"{count:,}\n({share:.1%})",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
            xytext=(0, 5),
            textcoords="offset points",
        )

    ax.set_title("Churn Distribution", fontsize=14, weight="bold")
    ax.set_xlabel("Customer Status")
    ax.set_ylabel("Customers")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_model_performance(y_test, probs, preds, output_path):
    sns.set_theme(style=PLOT_STYLE)
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].plot(fpr, tpr, color="#1f77b4", linewidth=2.5, label=f"AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#7f7f7f", linewidth=1)
    axes[0].set_title("ROC Curve", fontsize=14, weight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right", frameon=True)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=axes[1],
        xticklabels=["Predicted No", "Predicted Yes"],
        yticklabels=["Actual No", "Actual Yes"],
    )
    axes[1].set_title("Confusion Matrix", fontsize=14, weight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_feature_importance(pipeline, output_path):
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = None

    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
    elif hasattr(model, "coef_"):
        scores = abs(model.coef_[0])
    else:
        return

    if feature_names is None:
        feature_names = [f"feature_{index}" for index in range(len(scores))]

    fi = (
        pd.DataFrame({"feature": feature_names, "importance": scores})
        .sort_values("importance", ascending=False)
        .head(15)
        .sort_values("importance", ascending=True)
    )
    fi["feature"] = fi["feature"].map(_format_feature_name)

    sns.set_theme(style=PLOT_STYLE)
    plt.figure(figsize=(10, 6.5))
    ax = sns.barplot(data=fi, x="importance", y="feature", hue="feature", palette="Blues_r", legend=False)
    ax.set_title("Top Predictive Features", fontsize=14, weight="bold")
    ax.set_xlabel("Relative Importance")
    ax.set_ylabel("")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
