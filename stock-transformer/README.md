# Stock Price Forecasting with a Transformer

Final project for a deep learning course. It tests whether a small Transformer can forecast a stock's closing price, and whether an up/down classification framing works better.

## Data

Daily historical prices of a single stock (Microsoft) from a public Kaggle dataset. The notebook loads the CSV from Google Drive in Colab; place the file locally and change the path to run it elsewhere. The dataset is not included here.

## Method

1. EDA and seasonal decomposition, then an Augmented Dickey-Fuller stationarity check (with differencing).
2. Four Transformer setups:
   - S1: baseline regression.
   - S2: baseline plus hyperparameter search (units, dropout, learning rate).
   - S3: up/down classification, tuned, with SMOTE on the training windows.
   - S4: S3 plus an attention variant.

## Results

| Setup | Test result |
|---|---|
| S1 regression | RMSE 48.8, R² -2.83 |
| S2 regression, tuned (units 32, dropout 0.2, lr 0.001) | RMSE 89.7, R² -32.47 |
| S3 classification (223 test windows) | Accuracy 0.556, F1 0.715, recall of "up" 1.00, recall of "down" 0.00 |
| S4 classification | Same as S3: accuracy 0.556, F1 0.715 |

## What this shows

The negative R² values mean the regression models forecast worse than predicting the mean. In S3 and S4 the model predicts "up" for every window, which matches the share of "up" days (55.6%), so the F1 of 0.71 is the majority-class baseline and not evidence of skill. Tuning and SMOTE did not change that.

The takeaway from this project is the workflow (stationarity testing, windowing, tuning, an honest baseline comparison), not a usable forecasting model. Daily prices of a liquid stock are close to a random walk, and a small Transformer trained on price history alone does not find a signal in them.

## Limitations

- Single stock, single split, no walk-forward validation.
- Only price history is used; no other features.
- No LSTM or GRU comparison is included in this notebook.

## Files

- `FP_DeepL_Transformer_StockPredition.ipynb`
