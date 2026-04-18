# Business Analysis: Customer Churn Project

Customer retention is a major and important business topic for many industries and businesses. Losing customers reduces long-term growth opportunities and can increase marketing costs. This project analyzes customer churn behavior, identifies key churn drivers, builds predictive models, and provides business recommendations. The dataset is the Kaggle Telco Customer Churn dataset.

## Executive Summary

This project evaluates customer churn behavior using a reproducible analytics pipeline built on the Telco Customer Churn dataset. The analysis combines conservative data cleaning, baseline predictive modeling, and business interpretation to identify practical retention opportunities.

The final baseline model selected for this project is a logistic regression classifier. It achieved a test ROC AUC of `0.841` and a cross-validated ROC AUC of `0.846 +/- 0.012`, indicating that the model can distinguish churners from non-churners reasonably well for a first-pass business workflow.

The data also shows clear business patterns. Overall churn is `26.5%`, but risk is not evenly distributed across the customer base. Month-to-month customers churn at much higher rates than customers on annual or two-year contracts, and fiber optic customers appear to be the highest-risk internet service segment.

## Business Problem

Customer churn directly affects revenue stability, customer lifetime value, and marketing efficiency. Replacing lost customers usually costs more than retaining existing ones, so improving retention can produce both cost savings and stronger long-term growth.

For a telecom business, churn analysis supports decisions such as:

- which customers should be prioritized for retention outreach
- which contract types create higher churn exposure
- which service segments may require operational improvements
- where pricing, support, or onboarding strategies may be underperforming

## Analytical Approach

The project follows a simple and transparent workflow:

1. Load the raw churn data.
2. Clean the dataset conservatively without overwriting source files.
3. Standardize field names and data types.
4. Preserve limited missing values rather than dropping records aggressively.
5. Compare baseline classification models using cross-validated ROC AUC.
6. Select the strongest model and interpret the resulting churn signals.

This approach is appropriate for a business demo because it favors clarity, reproducibility, and understandable outputs over overly complex feature engineering.

## Data Quality Summary

The dataset contains `7,043` customers and `21` columns. Cleaning focused on basic structural issues:

- column names were standardized to `snake_case`
- text values were trimmed
- null-like strings were normalized to missing values
- known numeric fields were explicitly converted to numeric types

The only material remaining missingness is in `totalcharges`, where `11` rows (`0.2%`) are blank in the source data. Those rows were retained and handled through model-time imputation rather than removed.

The identifier column `customerid` was excluded from modeling to avoid leakage.

## Key Findings

### Churn Level

- Overall churn rate: `26.5%`

This means slightly more than one in four customers in the dataset left the service, which is high enough to justify focused retention action.

### Contract Risk

- Month-to-month churn rate: `42.7%`
- One-year churn rate: `11.3%`
- Two-year churn rate: `2.8%`

This is the clearest business signal in the project. Customers on flexible month-to-month plans are significantly more likely to churn than customers with longer commitments.

### Internet Segment Risk

- Fiber optic churn rate: `41.9%`
- DSL churn rate: `19.0%`
- No internet churn rate: `7.4%`

Fiber optic customers appear to be the most vulnerable segment. This may reflect pricing pressure, customer expectations, service quality issues, or a mix of those factors.

## Model Performance Summary

The selected baseline model was `logistic_regression`.

- Test ROC AUC: `0.841`
- Cross-validated ROC AUC: `0.846 +/- 0.012`
- Accuracy: `0.738`
- Precision: `0.504`
- Recall: `0.783`
- F1 score: `0.614`

From a business perspective, the relatively high recall is useful because it means the model captures a large share of likely churners. That makes it suitable for ranking customers for proactive retention campaigns, although the precision level suggests outreach should still be targeted and cost-aware.

## Business Recommendations

### 1. Prioritize Month-to-Month Retention

Month-to-month customers should be the first target for churn prevention. The business should test offers that make longer commitments more attractive, such as pricing incentives, bundled benefits, or loyalty discounts.

### 2. Investigate Fiber Optic Customer Experience

The fiber optic segment shows elevated churn and should be reviewed for possible root causes:

- service reliability
- support quality
- billing friction
- price sensitivity
- competitor pressure

### 3. Improve Early-Life Customer Engagement

Customers with short tenure are often more vulnerable to churn. A stronger onboarding process, better issue resolution in the first months, and earlier value communication could reduce avoidable losses.

### 4. Use Predictive Scores To Support Retention Operations

The churn model should be used as a prioritization layer for campaigns, account management, or customer success outreach. It is best treated as decision support rather than a fully automated action engine.

## Conclusion

This project demonstrates that even a simple, well-structured churn workflow can produce useful business insight. The strongest evidence points to contract structure and service segment as major drivers of churn risk. The project provides a solid starting point for retention strategy, model refinement, and future segmentation analysis.
