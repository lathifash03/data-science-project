# Fraud Detection on an Imbalanced Transaction Dataset

Final project for a machine learning course. The goal is to detect fraudulent mobile-money transactions when fraud is extremely rare.

## Data

- A 83,561-row sample of a PaySim-style synthetic mobile-money dataset (`step`, `type`, `amount`, origin and destination balances, `isFraud`).
- The data is synthetic, not real bank data. Put the CSV at `fraud_dataset.csv` (the notebook reads `/content/fraud_dataset.csv` in Colab).
- Fraud is 0.135% of rows: 90 fraud cases against 66,758 normal ones in the training split (ratio 1:741).

## Method

1. Exploratory analysis and missing-value check (one row has missing values).
2. Feature engineering: balance changes, amount-to-balance ratio, zero-balance flags, round-amount flag, hour of day, night and business-hours flags, amount z-score. Ten features with absolute correlation above 0.02 to the label were kept.
3. Stratified 80/20 split before any resampling. SMOTE (fraud raised to 10% of normal) is applied to the training set only. Robust scaling.
4. Scenario 1 (baseline): Logistic Regression and Decision Tree with class weights, tuned with 5-fold stratified cross-validation on ROC-AUC.

## Results (test set: 16,713 rows, 22 fraud cases)

| Model | Recall (fraud) | Precision (fraud) | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 1.000 | 0.016 | 0.031 | 0.991 | 0.312 |
| Decision Tree | 0.909 | 0.667 | 0.769 | 0.955 | 0.862 |

Logistic Regression catches every fraud case but raises 1,387 false alarms, so it is not usable as is. The Decision Tree gives the better trade-off. Accuracy is not informative here (a model that predicts "normal" for everything scores 99.87%).

## Limitations

- **Only 22 fraud cases in the test set.** One case changes recall by about 4.5 points, so these numbers have wide uncertainty. Treat the comparison between models as indicative.
- ROC-AUC looks high even for the weak Logistic Regression model. PR-AUC and F1 are the more honest metrics at this imbalance.
- The data is synthetic. In this kind of dataset the balance-change features almost encode how the fraud was generated, so results will not transfer to real transactions without re-validation.
- The notebook reports one random split with no repeated runs.

## Files

- `FinalProject_FraudDetection.ipynb`
