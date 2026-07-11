# BNPL Default Risk Analysis

Predicting customer default risk for Buy Now, Pay Later (BNPL) transactions, modeled on real-world fintech use cases like Tabby and Tamara.

## Business Problem

BNPL companies pay merchants the full purchase amount upfront, then collect repayment from customers in installments. When a customer defaults, the company absorbs the loss directly. This project builds a risk model that predicts which customers are likely to default or pay late, enabling smarter approve/review/reject decisions at checkout.

## Dataset

- 50,000 synthetic BNPL transactions, built to mimic real-world lending patterns (Kaggle, Apache License)
- 13 features: demographics (age, gender, income, credit score), transaction details (purchase amount, category, provider), and behavioral signals (device type, connection type, checkout time, browser)
- Target variable: `Repayment_Status` — Paid On Time (76.8%), Late Payment (14.7%), Defaulted (8.5%)
- Note: real BNPL transaction data isn't publicly available for privacy reasons, so this project uses a synthetic dataset designed to reflect realistic lending correlations.

## Key Challenge: Class Imbalance

An initial model trained without adjustment achieved 77% accuracy — but a 0% recall on defaulters. It had simply learned to predict "Paid On Time" for almost everyone, since that class dominates the data. This is a classic and costly failure mode for real risk models: a system that never flags risk is useless to a lending business.

**Fix:** retrained both models using `class_weight='balanced'`, forcing the model to weight rare outcomes (Defaulted, Late Payment) more heavily. This dropped overall accuracy to ~55%, but raised recall on Defaulted customers to 50%, meaning the model now actually catches half of all real defaulters — a direct, honest trade-off between accuracy and business usefulness.

## Models

| Model | Defaulted Recall | Defaulted Precision | Overall Accuracy |
|---|---|---|---|
| Logistic Regression (balanced) | 0.59 | 0.15 | 0.54 |
| Random Forest (balanced) | 0.50 | 0.21 | 0.55 |

Random Forest was selected as the primary model for its stronger balance of precision and recall across all three classes, plus interpretable feature importance output.

## Key Findings

- **Credit Score is the dominant risk driver** (43% feature importance), more than triple the next strongest factor.
- **Connection Type and Checkout Time** — unconventional, behavioral signals — ranked among the top predictors, suggesting checkout behavior carries real predictive signal beyond financial profile alone.
- **Travel and Electronics purchases carry the highest default risk**; Fashion purchases are the safest category.
- **Default risk peaks around age 30** and generally declines with age, likely reflecting income and credit history stability.
- **Risk is evenly distributed across BNPL providers** (Klarna, Affirm, Sezzle, Afterpay) — risk comes from the customer, not the provider.

## Risk Segmentation

Model outputs were converted into three actionable business tiers:

| Segment | Customers | Default Probability |
|---|---|---|
| Low Risk | 22,239 (44%) | < 0.15 |
| Medium Risk | 13,699 (27%) | 0.15 – 0.35 |
| High Risk | 14,062 (28%) | > 0.35 |

These thresholds are a starting point, not a tuned production system — in practice, a risk team would refine them against the company's acceptable false-positive rate.

## Dashboard

A 3-page Power BI dashboard translates the analysis into a business-ready tool:
1. **Overview** — portfolio KPIs, repayment distribution, category-level risk
2. **Risk Segmentation** — customer risk tiers, credit score vs. risk relationship, provider comparison
3. **Model Drivers** — feature importance, age-risk trend

See `/Screenshots` for dashboard previews.

## Limitations

- Synthetic data may not fully capture real-world BNPL customer behavior.
- Precision on the Defaulted class remains low (0.21), meaning the model currently favors catching risk over minimizing false alarms — a deliberate trade-off, but one that would need further tuning in production.
- Risk thresholds (0.15 / 0.35) were set heuristically, not optimized against real business costs.
- No time-series or repeat-customer behavior is captured, since the data is transaction-level, not longitudinal.
- XGBoost/LightGBM were not tested here; Logistic Regression and Random Forest were prioritized for interpretability, a priority in regulated lending contexts.

## Tech Stack

Python (Pandas, Scikit-learn, Matplotlib, Seaborn) · Power BI (DAX, Power Query) · Google Colab

## Files

- `BNPL.ipynb` — full analysis notebook
- `BNPL.pbix` — Power BI dashboard
- `BNPL_dataset_v2.csv` — source data
- `BNPL_results_for_powerbi.csv` — model output with predictions and risk segments
- `/Screenshots` — dashboard page previews
