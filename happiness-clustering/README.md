# Resident Happiness Survey: KNN and K-Means

Course assignment on a city resident satisfaction survey. It looks at which aspects of city life relate to a resident being happy, classifies happiness with KNN, and segments residents with K-Means.

## Data

A survey file (`happiness-classification.csv`) with ratings on city-service questions such as information availability, housing cost, school quality, trust in the local police, street maintenance, and events, plus a binary happiness label. The CSV is not included; the notebook reads it from `../content/` in Colab.

## Method

1. EDA of the answer distributions per question.
2. KNN classifier: 70/30 split, k chosen by tuning, final model with k = 3.
3. K-Means: elbow (WCSS) and silhouette score to choose the number of clusters (2 to 7 tested).

## Results

| Task | Result |
|---|---|
| KNN (67 test samples) | Accuracy 0.70; precision and recall 0.67 (unhappy) and 0.73 (happy) |
| K-Means | Silhouette score about 0.17 to 0.20 for 4 to 7 clusters; final model uses 2 clusters |

## Limitations

- The test set has only 67 samples, so the accuracy estimate is rough (one sample is about 1.5 points).
- Silhouette scores around 0.2 mean the clusters overlap heavily; the segmentation is exploratory, not a firm grouping of residents.
- Survey answers are ordinal scales treated as numeric.

## Files

- `Tugas2_Happines.ipynb`
