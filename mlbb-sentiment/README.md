# Sentiment Analysis of Mobile Legends Reviews (Google Play, Indonesian)

Final project for a course on machine learning. It scrapes Indonesian-language Google Play reviews of Mobile Legends: Bang Bang and trains classifiers to predict review sentiment.

## Data

- 108,000 reviews collected with `google-play-scraper` (language `id`, country `id`). Raw reviews are not included in this repository; run the scraping cell to collect your own copy.
- Star ratings are skewed negative: 54,143 one-star reviews against 23,196 five-star ones.

## Method

1. Cleaning: remove non-alphanumeric characters, lowercase, tokenization, stopword removal, Sastrawi stemming.
2. Labels: positive, neutral, or negative, assigned from the VADER compound score.
3. Features: TF-IDF.
4. Class balancing with SMOTE, then Random Forest, SVM, and Multinomial Naive Bayes, with the train/test ratio varied from 80/20 to 20/80.

## Results

- A Random Forest run with a 90/10 split reached 0.883 accuracy against the VADER labels.
- Across train/test ratios in the comparison table, accuracy was about 52% to 71%: Random Forest highest (61% to 71%), SVM lowest (52% to 59%), Naive Bayes in between. The values in that table were entered by hand in the notebook from earlier runs.

## Limitations (please read before reusing the numbers)

- **The labels are weak.** VADER is an English-language lexicon and these reviews are Indonesian, so most words are unknown to it. In the output, "optimalisasi gameplay yang sangat buruk" (very bad optimization) is labelled positive and "game nya bagus, grafiknya juga oke" (good game, good graphics) is labelled negative. Accuracy against these labels measures agreement with VADER, not real sentiment.
- SMOTE is applied before the train/test split in the 0.883 run, so synthetic samples can leak into the test set and inflate the score.
- No human-labelled sample was used to validate the labels.
- A better version would label with an Indonesian sentiment lexicon or a manually checked sample, apply SMOTE only on the training set, and report per-class F1.

## Files

- `FP_MCI_SentimentML.ipynb`
